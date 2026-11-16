from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ingestao
from src import preprocessamento as pp
from src.utils import load_config

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
