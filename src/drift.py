"""Monitoramento de data drift por Population Stability Index (PSI). Criterio 10.

PSI = sum((p_atual - p_ref) * ln(p_atual / p_ref)) sobre faixas definidas na referencia.
Faixas: quantis da referencia para atributos continuos; cada valor distinto para atributos
discretos (ate `drift.bins` valores). Limiares usuais de risco de credito: < 0.1 estavel,
0.1 a 0.25 moderado, >= 0.25 severo.

Referencia = lotes com que o modelo vigente foi treinado (registrados no manifesto).
Atual = lotes publicados depois disso. O status e decidido pelos atributos que o modelo
vigente consome, e um drift severo flexibiliza a regra de promocao (ver `promocao.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import CycleContext, read_json, resolve_path, write_json

EPS = 1e-4
SEVERITY = ["estavel", "moderado", "severo"]
NO_REFERENCE = "sem_referencia"
NO_NEW_DATA = "sem_dados_novos"


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().to_numpy(dtype=float), current.dropna().to_numpy(dtype=float)
    values = np.unique(np.concatenate([ref, cur]))
    if len(values) <= bins:
        ref_p = np.array([(ref == v).mean() for v in values])
        cur_p = np.array([(cur == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
        ref_p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
        cur_p = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    ref_p, cur_p = np.clip(ref_p, EPS, None), np.clip(cur_p, EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def classify(value: float, cfg: dict[str, Any]) -> str:
    t = cfg["drift"]["thresholds"]
    return "severo" if value >= t["severe"] else "moderado" if value >= t["moderate"] else "estavel"


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(reference[f], current[f], cfg["drift"]["bins"])} for f in features]
    table = pd.DataFrame(rows)
    table["status"] = table["psi"].map(lambda v: classify(v, cfg))
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def worst_status(table: pd.DataFrame, features: list[str]) -> str:
    statuses = table.loc[table["feature"].isin(features), "status"]
    return max(statuses, key=SEVERITY.index) if len(statuses) else "estavel"


@dataclass
class DriftResult:
    status: str
    reference_batches: list[str]
    current_batches: list[str]
    monitored_features: list[str]
    table: pd.DataFrame

    @property
    def severe(self) -> bool:
        return self.status == "severo"

    def to_dict(self, ctx: CycleContext) -> dict[str, Any]:
        return {**ctx.tags, "status": self.status, "severe": self.severe,
                "reference_batches": self.reference_batches, "current_batches": self.current_batches,
                "monitored_features": self.monitored_features, "psi": self.table.to_dict(orient="records")}


def analyze_new_batches(dataset: pd.DataFrame, manifest: dict[str, Any] | None, candidates: list[str],
                        cfg: dict[str, Any]) -> DriftResult:
    """Compara os lotes do modelo vigente com os lotes que chegaram depois dele."""
    if manifest is None:
        return DriftResult(NO_REFERENCE, [], sorted(dataset["lote"].unique()), [], pd.DataFrame(columns=["feature", "psi", "status"]))
    ref_batches = sorted(Path(b).stem for b in manifest["batches"])
    in_ref = dataset["lote"].isin(ref_batches)
    cur_batches = sorted(dataset.loc[~in_ref, "lote"].unique())
    if not cur_batches:
        return DriftResult(NO_NEW_DATA, ref_batches, [], manifest["selected_features"], pd.DataFrame(columns=["feature", "psi", "status"]))
    table = drift_table(dataset[in_ref], dataset[~in_ref], candidates, cfg)
    monitored = manifest["selected_features"]
    return DriftResult(worst_status(table, monitored), ref_batches, cur_batches, monitored, table)


def drift_path(cfg: dict[str, Any], root: Path | None = None) -> Path:
    return resolve_path(cfg, "processed_dir", root) / "drift.json"


def save_drift(result: DriftResult, ctx: CycleContext, cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    payload = result.to_dict(ctx)
    write_json(drift_path(cfg, root), payload)
    write_json(ctx.evidence_dir(cfg, root) / "01_drift.json", payload)
    return payload


def load_drift(cfg: dict[str, Any], ctx: CycleContext, root: Path | None = None) -> dict[str, Any]:
    payload = read_json(drift_path(cfg, root))
    if (payload["cycle_id"], payload["data_version"]) != (ctx.cycle_id, ctx.data_version):
        raise RuntimeError(f"Relatorio de drift e do ciclo {payload['cycle_id']}, nao de {ctx.cycle_id}. Rode o notebook 01.")
    return payload


def log_drift(result: DriftResult, ctx: CycleContext, cfg: dict[str, Any], root: Path | None = None) -> None:
    import mlflow

    from src.utils import cycle_run

    with cycle_run(ctx, cfg, stage="drift", root=root):
        mlflow.set_tags({"drift_status": result.status, "drift_reference": ",".join(result.reference_batches),
                         "drift_current": ",".join(result.current_batches)})
        mlflow.log_params({"drift_bins": cfg["drift"]["bins"], **{f"drift_threshold_{k}": v for k, v in cfg["drift"]["thresholds"].items()}})
        if not result.table.empty:
            mlflow.log_metrics({f"psi_{r.feature}": r.psi for r in result.table.itertuples()})
            mlflow.log_metric("psi_max_monitorado", float(result.table.loc[result.table["feature"].isin(result.monitored_features), "psi"].max()))
            mlflow.log_table(result.table, "drift_psi.json")
