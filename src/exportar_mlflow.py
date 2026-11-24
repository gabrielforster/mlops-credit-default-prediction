"""Exporta o historico do MLflow para evidencias legiveis sem abrir a interface.

Le `mlruns/mlflow.db` somente para leitura (nao reescreve caminhos de artefato) e grava:
- `evidencias/mlflow/runs.csv`: um run por linha, com tags e metricas;
- `evidencias/mlflow/runs.md`: arvore ciclo -> etapas com as metricas principais.

Uso: python -m src.exportar_mlflow
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.utils import load_config, project_root, resolve_path

TAGS = ["cycle_id", "data_version", "stage", "lotes", "algorithm", "winner", "drift_status",
        "promotion_decision", "active_model_id", "pipeline_status", "github_run_id"]
METRICS_MD = {
    "treino": ["val_f1", "val_precision", "val_recall", "val_roc_auc"],
    "promocao": ["active_test_f1", "active_test_roc_auc"],
    "votacao_features": ["n_elected"],
    "previsoes": ["reproducible", "taxa_prevista_inadimplencia"],
    "etl": ["linhas_treino", "linhas_validacao", "linhas_teste"],
}


def load_runs(db: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        runs = con.execute(
            "select r.run_uuid, r.name, r.status, r.start_time, e.name from runs r "
            "join experiments e on r.experiment_id = e.experiment_id order by r.start_time"
        ).fetchall()
        tags: dict[str, dict] = {}
        for rid, key, value in con.execute("select run_uuid, key, value from tags"):
            tags.setdefault(rid, {})[key] = value
        metrics: dict[str, dict] = {}
        for rid, key, value in con.execute("select run_uuid, key, value from latest_metrics"):
            metrics.setdefault(rid, {})[key] = value
    finally:
        con.close()
    out = []
    for rid, name, status, start, experiment in runs:
        t = tags.get(rid, {})
        out.append({
            "run_id": rid, "run_name": name, "status": status, "experiment": experiment,
            "start_utc": datetime.fromtimestamp(start / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "parent_run_id": t.get("mlflow.parentRunId", ""),
            **{k: t.get(k, "") for k in TAGS},
            "metrics": metrics.get(rid, {}),
        })
    return out


def write_csv(runs: list[dict], path: Path) -> None:
    metric_keys = sorted({k for r in runs for k in r["metrics"]})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        base = [k for k in runs[0] if k != "metrics"]
        writer.writerow(base + metric_keys)
        for r in runs:
            writer.writerow([r[k] for k in base] + [r["metrics"].get(k, "") for k in metric_keys])


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}"


def write_md(runs: list[dict], path: Path) -> None:
    lines = ["# Historico do MLflow", "",
             "Gerado por `python -m src.exportar_mlflow` a partir de `mlruns/mlflow.db`.",
             f"Experimento: `{runs[0]['experiment']}`. Runs: {len(runs)}.", ""]
    parents = [r for r in runs if not r["parent_run_id"]]
    for p in parents:
        lines += [f"## {p['cycle_id']}", "",
                  f"- run pai: `{p['run_id']}` ({p['start_utc']} UTC, {p['status']})",
                  f"- data_version: `{p['data_version']}` · lotes: {p['lotes']}",
                  f"- pipeline_status: `{p['pipeline_status']}` · decisao: `{p['promotion_decision']}`"
                  f" · vigente: `{p['active_model_id']}`", ""]
        if p["github_run_id"]:
            lines.insert(-1, f"- GitHub Actions run: `{p['github_run_id']}`")
        lines += ["| Etapa | run_id | Metricas |", "|---|---|---|"]
        for c in (r for r in runs if r["parent_run_id"] == p["run_id"]):
            keys = METRICS_MD.get(c["stage"], [])
            shown = ", ".join(f"{k}={_fmt(c['metrics'][k])}" for k in keys if k in c["metrics"])
            if c["stage"] == "drift":
                psi = {k: v for k, v in c["metrics"].items() if k.startswith("psi_")}
                shown = f"status={c['drift_status'] or 'sem_referencia'}"
                if psi:
                    top = max(psi, key=psi.get)
                    shown += f", maior {top}={psi[top]:.4f}"
            lines.append(f"| {c['run_name']} | `{c['run_id']}` | {shown} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = project_root()
    cfg = load_config(root)
    db = resolve_path(cfg, "mlruns_dir", root) / "mlflow.db"
    out = root / "evidencias" / "mlflow"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(db)
    write_csv(runs, out / "runs.csv")
    write_md(runs, out / "runs.md")
    print(f"{len(runs)} runs exportados para {out.relative_to(root)}")


if __name__ == "__main__":
    main()
