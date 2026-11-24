"""Selecao de atributos por tres metodos independentes e votacao.

Cada metodo devolve uma linha por atributo: `feature`, `score`, `rank` (1 = melhor) e `selected`.
A votacao elege os atributos selecionados por pelo menos `features.voting.minimum_votes` metodos e
garante `features.voting.minimum_features` atributos, desempatando pelo menor rank medio.
Todos os metodos sao ajustados apenas na particao de treino.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import CycleContext, read_json, resolve_path, write_json

SelectorFn = Callable[[pd.DataFrame, pd.Series, dict[str, Any]], pd.DataFrame]


def _ranked(features: list[str], score: np.ndarray, rank: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"feature": features, "score": score, "rank": rank.astype(int), "selected": selected.astype(bool)})


def _rank_desc(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(ascending=False, method="first").to_numpy()


def select_k_best(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Metodo 1: ANOVA F (`f_classif`) entre cada atributo e a classe. Mantem os k maiores F."""
    selector = SelectKBest(f_classif, k=cfg["features"]["k_best"]).fit(X, y)
    score = np.nan_to_num(selector.scores_)
    return _ranked(list(X.columns), score, _rank_desc(score), selector.get_support())


def random_forest_importance(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Metodo 2: importancia por reducao de impureza de uma RandomForest. Mantem as k maiores."""
    params = cfg["features"]["selector_params"]["random_forest"]
    forest = RandomForestClassifier(**params, random_state=cfg["data"]["random_state"]).fit(X, y)
    rank = _rank_desc(forest.feature_importances_)
    return _ranked(list(X.columns), forest.feature_importances_, rank, rank <= cfg["features"]["random_forest_features"])


def rfe_logistic(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Metodo 3: RFE com Regressao Logistica. Remove iterativamente o atributo de menor |coeficiente|."""
    params = cfg["features"]["selector_params"]["rfe_logistic_regression"]
    estimator = LogisticRegression(**params, random_state=cfg["data"]["random_state"])
    rfe = RFE(estimator, n_features_to_select=cfg["features"]["rfe_features"]).fit(X, y)
    # score: |coeficiente| final para os mantidos, 0 para os eliminados; rank = ordem de eliminacao do RFE.
    score = np.zeros(X.shape[1])
    score[rfe.support_] = np.abs(rfe.estimator_.coef_[0])
    return _ranked(list(X.columns), score, rfe.ranking_, rfe.support_)


SELECTORS: dict[str, SelectorFn] = {
    "select_k_best_f_classif": select_k_best,
    "random_forest_importance": random_forest_importance,
    "rfe_logistic_regression": rfe_logistic,
}


def selection_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Imputa e padroniza para a selecao. Categoricas entram como codigo inteiro apenas nesta triagem."""
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
    return pd.DataFrame(prep.transform(X), columns=X.columns, index=X.index)


def tally_votes(results: pd.DataFrame, minimum_votes: int, minimum_features: int = 1) -> pd.DataFrame:
    """Agrega o resultado longo (metodo x atributo) em votos por atributo, preservando a ordem original.

    Regra: eleito com `votes >= minimum_votes`. Se isso eleger menos que `minimum_features`, o conjunto
    e completado pelos atributos de mais votos; o desempate e o menor rank medio entre os metodos
    (`elected_by = "minimo"`). Empate restante: ordem original das colunas.
    """
    if not 1 <= minimum_votes <= results["method"].nunique():
        raise ValueError(f"minimum_votes={minimum_votes} fora de [1, {results['method'].nunique()}]")
    order = list(dict.fromkeys(results["feature"]))
    if not 1 <= minimum_features <= len(order):
        raise ValueError(f"minimum_features={minimum_features} fora de [1, {len(order)}]")
    grouped = results.groupby("feature")
    tally = pd.DataFrame({
        "feature": order,
        "votes": grouped["selected"].sum().reindex(order).astype(int).to_numpy(),
        "mean_rank": grouped["rank"].mean().reindex(order).to_numpy(),
    })
    tally["elected"] = tally["votes"] >= minimum_votes
    tally["elected_by"] = np.where(tally["elected"], "votos", "")
    missing = minimum_features - int(tally["elected"].sum())
    if missing > 0:
        fill = (tally[~tally["elected"]].assign(_pos=lambda d: range(len(d)))
                .sort_values(["votes", "mean_rank", "_pos"], ascending=[False, True, True]).index[:missing])
        tally.loc[fill, ["elected", "elected_by"]] = [True, "minimo"]
    return tally


@dataclass(frozen=True)
class VotingResult:
    records: pd.DataFrame
    tally: pd.DataFrame
    elected: list[str]


def run_feature_voting(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict[str, Any], ctx: CycleContext) -> VotingResult:
    """Executa os tres metodos sobre o treino e aplica a regra de votacao."""
    Xs = selection_matrix(X_train)
    per_method = [fn(Xs, y_train, cfg).assign(method=name) for name, fn in SELECTORS.items()]
    long = pd.concat(per_method, ignore_index=True)
    voting = cfg["features"]["voting"]
    tally = tally_votes(long, voting["minimum_votes"], voting["minimum_features"])
    elected = tally.loc[tally["elected"], "feature"].tolist()
    records = long.merge(tally[["feature", "votes", "mean_rank", "elected", "elected_by"]], on="feature").rename(
        columns={"selected": "vote", "elected": "final_selection"})
    records.insert(0, "data_version", ctx.data_version)
    records.insert(0, "cycle_id", ctx.cycle_id)
    records = records[["cycle_id", "data_version", "method", "feature", "score", "rank", "vote", "votes", "mean_rank",
                       "final_selection", "elected_by"]]
    return VotingResult(records=records, tally=tally, elected=elected)


def elected_features_path(cfg: dict[str, Any], root: Path | None = None) -> Path:
    return resolve_path(cfg, "processed_dir", root) / "features_eleitas.json"


def save_voting(result: VotingResult, ctx: CycleContext, cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    payload = {**ctx.tags, "minimum_votes": cfg["features"]["voting"]["minimum_votes"],
               "minimum_features": cfg["features"]["voting"]["minimum_features"],
               "tie_break": "mean_rank", "methods": list(SELECTORS), "elected_features": result.elected,
               "elected_by_minimum": result.tally.loc[result.tally["elected_by"] == "minimo", "feature"].tolist(),
               "votes": dict(zip(result.tally["feature"], result.tally["votes"].astype(int).tolist()))}
    write_json(elected_features_path(cfg, root), payload)
    evidence = ctx.evidence_dir(cfg, root)
    evidence.mkdir(parents=True, exist_ok=True)
    result.records.to_csv(evidence / "02_votacao_features.csv", index=False)
    write_json(evidence / "02_votacao_features.json", payload)
    return payload


def load_elected_features(cfg: dict[str, Any], ctx: CycleContext, root: Path | None = None) -> list[str]:
    """Carrega os atributos eleitos e garante que pertencem ao ciclo corrente."""
    payload = read_json(elected_features_path(cfg, root))
    if (payload["cycle_id"], payload["data_version"]) != (ctx.cycle_id, ctx.data_version):
        raise RuntimeError(f"Atributos eleitos sao do ciclo {payload['cycle_id']}, nao de {ctx.cycle_id}. Rode o notebook 02.")
    return payload["elected_features"]
