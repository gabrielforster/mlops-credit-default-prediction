"""Decisao de promocao, versionamento de modelos e manifesto do modelo vigente.

Regra (direcao `maximize`):  metrica_desafiante >= metrica_vigente + minimum_improvement
Regra (direcao `minimize`):  metrica_desafiante <= metrica_vigente - minimum_improvement

As duas metricas sao medidas na MESMA particao de validacao do ciclo corrente: o modelo
vigente e reavaliado nela. Sem modelo vigente, o vencedor e promovido como modelo inicial.
Cada modelo promovido ganha um diretorio proprio em `models/versoes/` que nunca e sobrescrito.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from src.modelagem import is_better
from src.utils import CycleContext, append_jsonl, read_json, resolve_path, sha256_file, utc_now, write_json

PROMOTED_INITIAL = "promovido_inicial"
PROMOTED = "promovido"
KEPT = "mantido"
DISABLED = "promocao_desabilitada"


@dataclass(frozen=True)
class PromotionDecision:
    decision: str
    challenger_metric: float
    champion_metric: float | None
    minimum_improvement: float
    threshold_relaxed_by_drift: bool
    reason: str

    @property
    def promoted(self) -> bool:
        return self.decision in (PROMOTED, PROMOTED_INITIAL)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "promoted": self.promoted}


def decide_promotion(challenger_metric: float, champion_metric: float | None, cfg: dict[str, Any],
                     severe_drift: bool = False) -> PromotionDecision:
    """Funcao pura: aplica a regra de promocao configurada."""
    metric, direction = cfg["evaluation"]["primary_metric"], cfg["evaluation"]["direction"]
    margin = float(cfg["promotion"]["minimum_improvement"])
    relaxed = bool(severe_drift and cfg["drift"]["relax_promotion_on_severe"] and champion_metric is not None)
    if relaxed:
        margin = 0.0
    if champion_metric is None:
        return PromotionDecision(PROMOTED_INITIAL, challenger_metric, None, margin, False,
                                 "nao existe modelo vigente: o vencedor do ciclo vira o modelo inicial")
    if not cfg["promotion"]["enabled"]:
        return PromotionDecision(DISABLED, challenger_metric, champion_metric, margin, False,
                                 "promotion.enabled = false: o modelo vigente e mantido")
    sign = "+" if direction == "maximize" else "-"
    rule = f"{metric}_desafiante {challenger_metric:.4f} vs {metric}_vigente {champion_metric:.4f} {sign} {margin:.4f}"
    if is_better(challenger_metric, champion_metric, direction, margin):
        return PromotionDecision(PROMOTED, challenger_metric, champion_metric, margin, relaxed, f"aprovado: {rule}")
    return PromotionDecision(KEPT, challenger_metric, champion_metric, margin, relaxed, f"reprovado: {rule}")


def model_id_for(ctx: CycleContext, algorithm: str) -> str:
    return f"{ctx.cycle_id}__{algorithm}"


def build_manifest(*, model_id: str, run_id: str | None, model_uri: str | None, ctx: CycleContext, algorithm: str,
                   model_label: str, cfg: dict[str, Any], metric_value: float, test_metrics: dict[str, float],
                   features: list[str], params: dict[str, Any], decision: PromotionDecision) -> dict[str, Any]:
    """Campos do manifesto. `artifact_path` e `artifact_sha256` sao adicionados ao gravar o artefato."""
    return {
        "model_id": model_id,
        "run_id": run_id,
        "model_uri": model_uri,
        "cycle_id": ctx.cycle_id,
        "data_version": ctx.data_version,
        "batches": ctx.batches,
        "algorithm": algorithm,
        "model_name": model_label,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "direction": cfg["evaluation"]["direction"],
        "metric_value": metric_value,
        "metric_partition": "validacao",
        "test_metrics": test_metrics,
        "selected_features": features,
        "hyperparameters": params,
        "random_state": cfg["data"]["random_state"],
        "promotion_decision": decision.decision,
        "decision_reason": decision.reason,
        "timestamp": utc_now(),
    }


def load_manifest(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Any] | None:
    path = resolve_path(cfg, "manifest", root)
    return read_json(path) if path.is_file() else None


def load_active_model(cfg: dict[str, Any], root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Carrega o pacote do modelo vigente (pipeline + atributos) e confere integridade contra o manifesto."""
    manifest = load_manifest(cfg, root)
    if manifest is None:
        return None
    from src.utils import project_root

    artifact = (root or project_root()) / manifest["artifact_path"]
    if sha256_file(artifact) != manifest["artifact_sha256"]:
        raise RuntimeError(f"Artefato {artifact} nao confere com o SHA-256 do manifesto")
    bundle = joblib.load(artifact)
    if bundle["features"] != manifest["selected_features"]:
        raise RuntimeError("Atributos do pacote divergem dos atributos do manifesto")
    return bundle, manifest


def persist_promoted_model(bundle: dict[str, Any], manifest_fields: dict[str, Any], cfg: dict[str, Any],
                           root: Path | None = None) -> dict[str, Any]:
    """Grava a nova versao em `models/versoes/<model_id>/` (sem sobrescrever) e atualiza o manifesto."""
    from src.utils import project_root

    root = root or project_root()
    version_dir = resolve_path(cfg, "models_dir", root) / "versoes" / bundle["model_id"]
    if version_dir.exists():
        raise FileExistsError(f"Versao {bundle['model_id']} ja existe; versoes anteriores nunca sao sobrescritas")
    version_dir.mkdir(parents=True)
    artifact = version_dir / "modelo.joblib"
    joblib.dump(bundle, artifact, compress=3)
    manifest = {**manifest_fields, "artifact_path": artifact.relative_to(root).as_posix(),
                "artifact_sha256": sha256_file(artifact)}
    write_json(version_dir / "manifesto.json", manifest)
    write_json(resolve_path(cfg, "manifest", root), manifest)
    return manifest


def record_decision(entry: dict[str, Any], ctx: CycleContext, cfg: dict[str, Any], root: Path | None = None) -> None:
    """Toda decisao (promovida ou nao) entra no historico e nas evidencias do ciclo."""
    append_jsonl(resolve_path(cfg, "promotion_history", root), entry)
    write_json(ctx.evidence_dir(cfg, root) / "04_promocao.json", entry)
