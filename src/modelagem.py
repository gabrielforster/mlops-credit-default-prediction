"""Treino e competicao entre o modelo A (LogisticRegression) e o modelo B (RandomForestClassifier).

Os dois modelos recebem os mesmos atributos eleitos, o mesmo treino, a mesma validacao e a
mesma metrica primaria. Cada modelo e um `Pipeline(pre-processador, estimador)`, entao o
pre-processamento e ajustado so no treino e viaja junto com o modelo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessamento import build_preprocessor
from src.utils import CycleContext, read_json, resolve_path, write_json

ALGORITHMS: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

# Cada metrica recebe (y_true, y_pred, y_score).
METRICS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "f1": lambda y, p, s: f1_score(y, p, zero_division=0),
    "precision": lambda y, p, s: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, s: recall_score(y, p, zero_division=0),
    "accuracy": lambda y, p, s: accuracy_score(y, p),
    "roc_auc": lambda y, p, s: roc_auc_score(y, s),
}


# O MLflow 3 serializa com skops, que exige declarar tipos fora da lista segura padrao.
# Os modelos sao gerados por este proprio pipeline, entao esses tipos sao confiaveis.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def metric_names(cfg: dict[str, Any]) -> list[str]:
    primary = cfg["evaluation"]["primary_metric"]
    return [primary, *[m for m in cfg["evaluation"]["secondary_metrics"] if m != primary]]


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> dict[str, float]:
    """Metrica primaria, secundarias e matriz de confusao (tn, fp, fn, tp)."""
    pred = pipeline.predict(X)
    score = pipeline.predict_proba(X)[:, 1]
    out = {name: float(METRICS[name](y, pred, score)) for name in metric_names(cfg)}
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {**out, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_model(algorithm: str, params: dict[str, Any], features: list[str], random_state: int) -> Pipeline:
    estimator = ALGORITHMS[algorithm](**params, random_state=random_state)
    return Pipeline([("preprocessador", build_preprocessor(features)), ("modelo", estimator)])


@dataclass
class ModelResult:
    algorithm: str
    label: str
    params: dict[str, Any]
    random_state: int
    features: list[str]
    pipeline: Pipeline
    validation: dict[str, float]
    train_seconds: float
    n_train: int
    n_validation: int
    run_id: str | None = None
    model_uri: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "label": self.label, "params": self.params,
                "random_state": self.random_state, "features": self.features, "validation": self.validation,
                "train_seconds": round(self.train_seconds, 3), "n_train": self.n_train,
                "n_validation": self.n_validation, "run_id": self.run_id, "model_uri": self.model_uri}


def enabled_models(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = {name: spec for name, spec in cfg["models"].items() if spec.get("enabled", True)}
    unknown = set(models) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"Algoritmos desconhecidos em config.models: {sorted(unknown)}")
    if len(models) != 2:
        raise ValueError(f"A competicao exige exatamente 2 modelos habilitados, ha {len(models)}")
    return models


def train_and_evaluate(algorithm: str, spec: dict[str, Any], features: list[str],
                       train: tuple[pd.DataFrame, pd.Series], valid: tuple[pd.DataFrame, pd.Series],
                       cfg: dict[str, Any]) -> ModelResult:
    seed = cfg["data"]["random_state"]
    pipeline = build_model(algorithm, spec.get("params", {}), features, seed)
    start = time.perf_counter()
    pipeline.fit(train[0][features], train[1])
    elapsed = time.perf_counter() - start
    return ModelResult(algorithm=algorithm, label=spec.get("label", algorithm), params=dict(spec.get("params", {})),
                       random_state=seed, features=list(features), pipeline=pipeline,
                       validation=evaluate(pipeline, valid[0][features], valid[1], cfg),
                       train_seconds=elapsed, n_train=len(train[0]), n_validation=len(valid[0]))


def is_better(candidate: float, reference: float, direction: str, margin: float = 0.0) -> bool:
    """`candidate` supera `reference` por pelo menos `margin` na direcao da metrica."""
    if direction == "maximize":
        return candidate >= reference + margin
    if direction == "minimize":
        return candidate <= reference - margin
    raise ValueError(f"direction invalida: {direction}")


def select_winner(results: list[ModelResult], cfg: dict[str, Any]) -> ModelResult:
    """Vence a melhor metrica primaria na validacao. Empate: vence o primeiro na ordem de `config.models`."""
    metric, direction = cfg["evaluation"]["primary_metric"], cfg["evaluation"]["direction"]
    winner = results[0]
    for challenger in results[1:]:
        if challenger.validation[metric] != winner.validation[metric] and \
                is_better(challenger.validation[metric], winner.validation[metric], direction):
            winner = challenger
    return winner


def log_model_run(result: ModelResult, ctx: CycleContext, cfg: dict[str, Any], voting: dict[str, Any],
                  X_example: pd.DataFrame, root: Path | None = None) -> None:
    """Registra um modelo como run filho do ciclo: parametros, metricas, atributos e artefato."""
    import mlflow
    from mlflow.models import infer_signature

    from src.utils import cycle_run

    with cycle_run(ctx, cfg, stage="treino", run_name=f"{result.label}_{result.algorithm}", root=root) as run:
        mlflow.set_tags({"model_name": result.label, "algorithm": result.algorithm,
                         "primary_metric": cfg["evaluation"]["primary_metric"]})
        mlflow.log_params({**{f"hp_{k}": v for k, v in result.params.items()},
                           "random_state": result.random_state, "algorithm": result.algorithm,
                           "model_name": result.label, "cycle_id": ctx.cycle_id, "data_version": ctx.data_version,
                           "n_features": len(result.features), "feature_selection_methods": ",".join(voting["methods"]),
                           "feature_selection_minimum_votes": voting["minimum_votes"],
                           "n_train": result.n_train, "n_validation": result.n_validation})
        mlflow.log_metrics({f"val_{k}": v for k, v in result.validation.items()})
        mlflow.log_metric("train_seconds", result.train_seconds)
        mlflow.log_dict({"selected_features": result.features}, "selected_features.json")
        sample = X_example[result.features].head(5)
        info = mlflow.sklearn.log_model(result.pipeline, name="modelo", input_example=sample,
                                        skops_trusted_types=SKOPS_TRUSTED_TYPES,
                                        signature=infer_signature(sample, result.pipeline.predict_proba(sample)[:, 1]))
        result.run_id, result.model_uri = run.info.run_id, info.model_uri


def competition_path(cfg: dict[str, Any], root: Path | None = None) -> Path:
    return resolve_path(cfg, "processed_dir", root) / "competicao.json"


def save_competition(results: list[ModelResult], winner: ModelResult, ctx: CycleContext, cfg: dict[str, Any],
                     root: Path | None = None) -> dict[str, Any]:
    payload = {**ctx.tags, "primary_metric": cfg["evaluation"]["primary_metric"],
               "direction": cfg["evaluation"]["direction"], "winner": winner.label,
               "models": [r.summary() for r in results]}
    write_json(competition_path(cfg, root), payload)
    write_json(ctx.evidence_dir(cfg, root) / "03_competicao.json", payload)
    return payload


def load_competition(cfg: dict[str, Any], ctx: CycleContext, root: Path | None = None) -> dict[str, Any]:
    payload = read_json(competition_path(cfg, root))
    if (payload["cycle_id"], payload["data_version"]) != (ctx.cycle_id, ctx.data_version):
        raise RuntimeError(f"Competicao registrada e do ciclo {payload['cycle_id']}, nao de {ctx.cycle_id}. Rode o notebook 03.")
    return payload
