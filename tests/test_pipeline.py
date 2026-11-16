from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import drift, ingestao, modelagem as mdl, promocao
from src import features as fs
from src import preprocessamento as pp
from src.utils import CycleContext, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT)


def synthetic_raw(cfg, n=200, seed=0) -> pd.DataFrame:
    """Dados no formato da fonte (alvo ja renomeado), todos dentro do dominio valido."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
        **{c: rng.integers(-2, 9, n) for c in pp.PAY_STATUS},
        **{c: rng.integers(-1_000, 100_000, n) for c in pp.BILL_AMT},
        **{c: rng.integers(0, 20_000, n) for c in pp.PAY_AMT},
    })
    df[cfg["data"]["target"]] = rng.integers(0, 2, n)
    return df


def test_validation_accepts_clean_data(cfg):
    report = pp.validate(synthetic_raw(cfg), cfg)
    assert report.is_valid
    assert report.invalid_values == {}


@pytest.mark.parametrize(
    "corrupt, expected",
    [
        (lambda df, t: df.drop(columns=["AGE"]), "colunas ausentes"),
        (lambda df, t: df.assign(ID=1), "IDs duplicados"),
        (lambda df, t: df.assign(**{t: 2}), "alvo com valores fora"),
        (lambda df, t: df.assign(**{t: np.nan}), "linhas sem rotulo"),
        (lambda df, t: df.assign(AGE="x"), "colunas nao numericas"),
    ],
)
def test_validation_rejects_structural_errors(cfg, corrupt, expected):
    report = pp.validate(corrupt(synthetic_raw(cfg), cfg["data"]["target"]), cfg)
    assert not report.is_valid
    assert any(expected in e for e in report.errors)


def test_validation_reports_missing_values_and_undocumented_codes(cfg):
    df = synthetic_raw(cfg)
    df.loc[0, "LIMIT_BAL"] = np.nan
    df.loc[1, "EDUCATION"] = 6
    report = pp.validate(df, cfg)
    assert report.missing_values == {"LIMIT_BAL": 1}
    assert report.undocumented_codes == {"EDUCATION": {6: 1}}


def test_clean_recodes_undocumented_codes_and_keeps_rows(cfg):
    df = synthetic_raw(cfg)
    df.loc[0, "EDUCATION"] = 0
    df.loc[1, "MARRIAGE"] = 0
    cleaned, log = pp.clean(df, cfg)
    assert len(cleaned) == len(df)
    assert cleaned.loc[0, "EDUCATION"] == 4 and cleaned.loc[1, "MARRIAGE"] == 3
    assert log["linhas_removidas"] == 0


def test_clean_fails_when_too_many_invalid_rows(cfg):
    df = synthetic_raw(cfg)
    df.loc[: len(df) // 10, "AGE"] = 5
    with pytest.raises(pp.DataValidationError):
        pp.clean(df, cfg)


def test_engineered_features_are_finite_and_row_local(cfg):
    df = synthetic_raw(cfg)
    full = pp.engineer_features(df)
    assert np.isfinite(full[pp.ENGINEERED].to_numpy(dtype=float)).all()
    single = pp.engineer_features(df.iloc[[7]])
    pd.testing.assert_frame_equal(single[pp.ENGINEERED].reset_index(drop=True),
                                  full.iloc[[7]][pp.ENGINEERED].reset_index(drop=True))


def test_split_is_stable_when_new_batches_arrive(cfg):
    ids = pd.Series(np.arange(1, 5_001))
    first = pp.assign_split(ids, cfg)
    grown = pp.assign_split(pd.Series(np.arange(1, 8_001)), cfg)
    assert (grown.iloc[:5_000].to_numpy() == first.to_numpy()).all()
    shares = first.value_counts(normalize=True)
    assert shares["teste"] == pytest.approx(cfg["evaluation"]["test_size"], abs=0.03)
    assert shares["validacao"] == pytest.approx(cfg["evaluation"]["validation_size"], abs=0.03)


def test_preprocessor_learns_only_from_fitted_data(cfg):
    df = pp.engineer_features(synthetic_raw(cfg))
    features = pp.feature_columns(cfg, df)
    train, other = df.iloc[:100], df.iloc[100:].assign(LIMIT_BAL=10_000_000)
    pre = pp.build_preprocessor(features).fit(train[features])
    pre.transform(other[features])
    numeric = list(pre.named_transformers_["num"].feature_names_in_)
    learned = pre.named_transformers_["num"]["scaler"].mean_[numeric.index("LIMIT_BAL")]
    assert learned == pytest.approx(train["LIMIT_BAL"].mean())


def test_partition_is_deterministic_and_disjoint(cfg):
    source = synthetic_raw(cfg, n=1_000).rename(columns={cfg["data"]["target"]: cfg["data"]["source"]["original_target"]})
    batches, inference = ingestao.partition_source(cfg, source)
    again, _ = ingestao.partition_source(cfg, source)
    ids = [set(b["ID"]) for b in batches] + [set(inference["ID"])]
    assert sum(len(s) for s in ids) == len(source)
    assert len(set().union(*ids)) == len(source)
    assert cfg["data"]["source"]["original_target"] not in inference.columns
    for a, b in zip(batches, again):
        pd.testing.assert_frame_equal(a, b)


def test_data_version_is_deterministic_and_content_addressed(tmp_path):
    a, b = tmp_path / "lote_001.csv", tmp_path / "lote_002.csv"
    a.write_text("ID\n1\n")
    b.write_text("ID\n2\n")
    v1 = ingestao.compute_data_version([a])
    assert v1 == ingestao.compute_data_version([a])
    assert v1 != ingestao.compute_data_version([a, b])
    a.write_text("ID\n3\n")
    assert v1 != ingestao.compute_data_version([a])


# --------------------------------------------------------------------------- votacao de atributos

def make_ctx(cycle_id="ciclo_teste", data_version="dv_teste", batches=("lote_001.csv",)) -> CycleContext:
    return CycleContext(cycle_id=cycle_id, data_version=data_version, started_at="2026-01-01T00:00:00+00:00",
                        batches=list(batches))


def votes_frame(selected_by_method: dict[str, set[str]], features=("a", "b", "c", "d")) -> pd.DataFrame:
    return pd.DataFrame([{"method": m, "feature": f, "selected": f in chosen}
                         for m, chosen in selected_by_method.items() for f in features])


@pytest.mark.parametrize("minimum_votes, expected", [(1, ["a", "b", "c"]), (2, ["a", "b"]), (3, ["a"])])
def test_tally_votes_applies_configurable_minimum(minimum_votes, expected):
    votes = votes_frame({"m1": {"a", "b"}, "m2": {"a", "b", "c"}, "m3": {"a"}})
    tally = fs.tally_votes(votes, minimum_votes)
    assert tally.loc[tally["elected"], "feature"].tolist() == expected
    assert dict(zip(tally["feature"], tally["votes"])) == {"a": 3, "b": 2, "c": 1, "d": 0}


def test_tally_votes_rejects_impossible_minimum():
    with pytest.raises(ValueError):
        fs.tally_votes(votes_frame({"m1": {"a"}, "m2": {"a"}, "m3": {"a"}}), 4)


@pytest.fixture(scope="module")
def train_frame(cfg):
    df = pp.engineer_features(synthetic_raw(cfg, n=600, seed=1))
    signal = (df["PAY_0"] >= 2) | (df["LIMIT_BAL"] < 60_000)
    df[cfg["data"]["target"]] = signal.astype(int)
    return df


def test_feature_voting_runs_three_methods_and_records_every_vote(cfg, train_frame):
    candidates = pp.feature_columns(cfg, train_frame)
    result = fs.run_feature_voting(train_frame[candidates], train_frame[cfg["data"]["target"]], cfg, make_ctx())
    rec = result.records
    assert list(rec.columns) == ["cycle_id", "data_version", "method", "feature", "score", "rank", "vote", "votes", "final_selection"]
    assert set(rec["method"]) == set(fs.SELECTORS) and len(rec) == 3 * len(candidates)
    per_method = rec.groupby("method")["vote"].sum().to_dict()
    assert per_method == {"select_k_best_f_classif": cfg["features"]["k_best"],
                          "random_forest_importance": cfg["features"]["random_forest_features"],
                          "rfe_logistic_regression": cfg["features"]["rfe_features"]}
    elected_from_records = rec.loc[rec["final_selection"], "feature"].unique().tolist()
    assert elected_from_records == result.elected
    assert {"PAY_0", "LIMIT_BAL"} <= set(result.elected)


# --------------------------------------------------------------------------- competicao entre modelos

def fake_result(label, f1):
    return mdl.ModelResult(algorithm=label, label=label, params={}, random_state=42, features=["x"], pipeline=None,
                           validation={"f1": f1}, train_seconds=0.0, n_train=1, n_validation=1)


@pytest.mark.parametrize("direction, scores, winner", [
    ("maximize", [0.50, 0.60], "B"),
    ("maximize", [0.60, 0.50], "A"),
    ("maximize", [0.55, 0.55], "A"),
    ("minimize", [0.50, 0.60], "A"),
])
def test_select_winner_respects_direction_and_tie_break(cfg, direction, scores, winner):
    local = {**cfg, "evaluation": {**cfg["evaluation"], "direction": direction}}
    results = [fake_result("A", scores[0]), fake_result("B", scores[1])]
    assert mdl.select_winner(results, local).label == winner


def test_both_models_train_on_the_same_features(cfg, train_frame):
    features = ["LIMIT_BAL", "PAY_0", "SEX", "UTILIZACAO_MEDIA"]
    target = train_frame[cfg["data"]["target"]]
    train, valid = (train_frame.iloc[:400], target.iloc[:400]), (train_frame.iloc[400:], target.iloc[400:])
    results = [mdl.train_and_evaluate(a, s, features, train, valid, cfg) for a, s in mdl.enabled_models(cfg).items()]
    assert [r.algorithm for r in results] == ["logistic_regression", "random_forest"]
    for r in results:
        assert list(r.pipeline.named_steps["preprocessador"].feature_names_in_) == features
        assert set(mdl.metric_names(cfg)) | {"tn", "fp", "fn", "tp"} <= set(r.validation)


# --------------------------------------------------------------------------- promocao

@pytest.mark.parametrize("challenger, champion, severe, expected", [
    (0.50, None, False, promocao.PROMOTED_INITIAL),
    (0.62, 0.60, False, promocao.PROMOTED),
    (0.61, 0.60, False, promocao.PROMOTED),
    (0.605, 0.60, False, promocao.KEPT),
    (0.55, 0.60, False, promocao.KEPT),
    (0.605, 0.60, True, promocao.PROMOTED),
    (0.59, 0.60, True, promocao.KEPT),
])
def test_promotion_rule(cfg, challenger, champion, severe, expected):
    decision = promocao.decide_promotion(challenger, champion, cfg, severe_drift=severe)
    assert decision.decision == expected
    assert decision.promoted == (expected in (promocao.PROMOTED, promocao.PROMOTED_INITIAL))


def test_promotion_disabled_keeps_champion(cfg):
    local = {**cfg, "promotion": {**cfg["promotion"], "enabled": False}}
    assert promocao.decide_promotion(0.99, 0.10, local).decision == promocao.DISABLED


def test_promotion_minimize_direction(cfg):
    local = {**cfg, "evaluation": {**cfg["evaluation"], "direction": "minimize"}}
    assert promocao.decide_promotion(0.30, 0.40, local).promoted
    assert not promocao.decide_promotion(0.395, 0.40, local).promoted


# --------------------------------------------------------------------------- manifesto e versionamento

def test_manifest_persistence_never_overwrites_and_detects_tampering(cfg, tmp_path, train_frame):
    ctx = make_ctx()
    features = ["LIMIT_BAL", "PAY_0"]
    target = train_frame[cfg["data"]["target"]]
    pipeline = mdl.build_model("logistic_regression", {"max_iter": 500}, features, 42).fit(train_frame[features], target)
    decision = promocao.decide_promotion(0.5, None, cfg)
    fields = promocao.build_manifest(model_id="m1", run_id="r1", model_uri="models:/x", ctx=ctx, algorithm="logistic_regression",
                                     model_label="modelo_A", cfg=cfg, metric_value=0.5, test_metrics={"f1": 0.4},
                                     features=features, params={"max_iter": 500}, decision=decision)
    bundle = {"pipeline": pipeline, "features": features, "model_id": "m1"}

    manifest = promocao.persist_promoted_model(bundle, fields, cfg, tmp_path)
    required = {"model_id", "run_id", "cycle_id", "data_version", "algorithm", "primary_metric", "metric_value",
                "selected_features", "promotion_decision", "timestamp", "artifact_path", "artifact_sha256"}
    assert required <= set(manifest)
    assert (manifest["cycle_id"], manifest["data_version"]) == (ctx.cycle_id, ctx.data_version)
    assert manifest["primary_metric"] == cfg["evaluation"]["primary_metric"]
    assert promocao.load_manifest(cfg, tmp_path) == manifest

    loaded, _ = promocao.load_active_model(cfg, tmp_path)
    np.testing.assert_array_equal(loaded["pipeline"].predict(train_frame[features]), pipeline.predict(train_frame[features]))

    with pytest.raises(FileExistsError):
        promocao.persist_promoted_model(bundle, fields, cfg, tmp_path)

    (tmp_path / manifest["artifact_path"]).write_bytes(b"corrompido")
    with pytest.raises(RuntimeError, match="SHA-256"):
        promocao.load_active_model(cfg, tmp_path)


# --------------------------------------------------------------------------- drift (criterio 10)

def test_psi_separates_same_population_from_shifted_population(cfg):
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 5_000))
    assert drift.classify(drift.psi(reference, pd.Series(rng.normal(0, 1, 5_000))), cfg) == "estavel"
    assert drift.classify(drift.psi(reference, pd.Series(rng.normal(1, 1, 5_000))), cfg) == "severo"


def test_psi_handles_discrete_features(cfg):
    ref = pd.Series([0] * 80 + [1] * 20)
    assert drift.psi(ref, ref.copy()) == pytest.approx(0.0)
    assert drift.classify(drift.psi(ref, pd.Series([0] * 40 + [1] * 60)), cfg) == "severo"


def test_drift_uses_champion_batches_as_reference(cfg):
    rng = np.random.default_rng(0)
    old = pd.DataFrame({"x": rng.normal(0, 1, 2_000), "y": rng.normal(0, 1, 2_000), "lote": "lote_001"})
    new = pd.DataFrame({"x": rng.normal(2, 1, 2_000), "y": rng.normal(0, 1, 2_000), "lote": "lote_002"})
    data = pd.concat([old, new], ignore_index=True)

    assert drift.analyze_new_batches(data, None, ["x", "y"], cfg).status == drift.NO_REFERENCE
    only_old = {"batches": ["lote_001.csv", "lote_002.csv"], "selected_features": ["x"]}
    assert drift.analyze_new_batches(data, only_old, ["x", "y"], cfg).status == drift.NO_NEW_DATA

    watching_x = drift.analyze_new_batches(data, {"batches": ["lote_001.csv"], "selected_features": ["x"]}, ["x", "y"], cfg)
    assert (watching_x.status, watching_x.reference_batches, watching_x.current_batches) == ("severo", ["lote_001"], ["lote_002"])
    watching_y = drift.analyze_new_batches(data, {"batches": ["lote_001.csv"], "selected_features": ["y"]}, ["x", "y"], cfg)
    assert watching_y.status == "estavel"
