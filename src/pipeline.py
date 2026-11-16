"""Executor do ciclo completo: roda os notebooks 01 a 05 em sequencia e para no primeiro erro.

Uso:
    python -m src.pipeline              # novo ciclo com os lotes ja publicados
    python -m src.pipeline --novo-lote  # publica o proximo lote (se houver) e roda o ciclo
    python -m src.pipeline --comparar   # apenas regenera reports/evidencias/comparacao_ciclos.md
    python -m src.pipeline --mlflow-ui  # abre a interface do MLflow sobre mlruns/ (http://127.0.0.1:5000)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src import ingestao
from src.utils import CycleContext, configure_mlflow, get_logger, load_config, load_cycle, project_root, read_json, resolve_path, write_json

log = get_logger(__name__)

NOTEBOOKS = [
    "01_etl.ipynb",
    "02_votacao_features.ipynb",
    "03_treino_competicao.ipynb",
    "04_selecao_promocao.ipynb",
    "05_previsoes.ipynb",
]
NOTEBOOK_TIMEOUT_SECONDS = 1800


def execute_notebook(path: Path) -> None:
    """Executa o notebook no proprio diretorio e grava as saidas no arquivo, mesmo em caso de falha."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=NOTEBOOK_TIMEOUT_SECONDS, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)


def current_cycle_id(cfg: dict[str, Any], root: Path) -> str | None:
    try:
        return load_cycle(cfg, root).cycle_id
    except FileNotFoundError:
        return None


def tag_cycle(ctx: CycleContext, cfg: dict[str, Any], root: Path, status: str) -> None:
    import mlflow

    configure_mlflow(cfg, root)
    mlflow.MlflowClient().set_tag(ctx.parent_run_id, "pipeline_status", status)


def cycle_summary(evidence: Path) -> dict[str, Any]:
    """Consolida as evidencias de cada etapa de um ciclo em um unico resumo."""
    def get(name: str) -> dict[str, Any]:
        return read_json(evidence / name) if (evidence / name).is_file() else {}

    etl, drift, voting = get("01_etl.json"), get("01_drift.json"), get("02_votacao_features.json")
    comp, promo, pred = get("03_competicao.json"), get("04_promocao.json"), get("05_previsoes.json")
    metric = comp.get("primary_metric", "f1")
    return {
        "cycle_id": etl.get("cycle_id"),
        "data_version": etl.get("data_version"),
        "lotes": promo.get("batches") or etl.get("lotes"),
        "n_linhas": etl.get("n_linhas"),
        "particoes": etl.get("particoes"),
        "drift_status": drift.get("status"),
        "atributos_eleitos": voting.get("elected_features"),
        "metrica_primaria": metric,
        "modelos": {m["label"]: {"algorithm": m["algorithm"], f"val_{metric}": m["validation"].get(metric),
                                 "val_roc_auc": m["validation"].get("roc_auc")} for m in comp.get("models", [])},
        "vencedor": comp.get("winner"),
        "decisao": promo.get("decision"),
        "motivo": promo.get("reason"),
        "metrica_desafiante": promo.get("challenger_metric"),
        "metrica_vigente_reavaliada": promo.get("champion_metric"),
        "modelo_vigente_apos_ciclo": promo.get("active_model_id"),
        "teste_modelo_vigente": promo.get("active_model_test_metrics"),
        "previsoes": pred.get("n_previsoes"),
        "previsoes_reproduziveis": pred.get("reproducible"),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return "-" if value is None else str(value)


def compare_cycles(cfg: dict[str, Any], root: Path) -> Path:
    """Gera `comparacao_ciclos.md`: uma coluna por ciclo concluido, uma linha por item avaliado."""
    evidence_root = resolve_path(cfg, "evidence_dir", root)
    summaries = [read_json(p) for p in sorted(evidence_root.glob("*/resumo_ciclo.json"))]
    rows = [("cycle_id", "cycle_id"), ("data_version", "data_version"), ("Lotes", "lotes"), ("Linhas", "n_linhas"),
            ("Drift (PSI)", "drift_status"), ("Atributos eleitos", "atributos_eleitos")]
    labels = sorted({label for s in summaries for label in s["modelos"]})
    lines = ["# Comparacao entre ciclos", "", "Gerado por `python -m src.pipeline`. Fonte: `reports/evidencias/<cycle_id>/`.", "",
             "| Item | " + " | ".join(f"Ciclo {i + 1}" for i in range(len(summaries))) + " |",
             "|---|" + "---|" * len(summaries)]
    table = [(name, [s.get(key) for s in summaries]) for name, key in rows]
    for label in labels:
        for metric_key in (f"val_{summaries[0]['metrica_primaria']}", "val_roc_auc"):
            table.append((f"{label} {metric_key}", [s["modelos"].get(label, {}).get(metric_key) for s in summaries]))
    table += [(name, [s.get(key) for s in summaries]) for name, key in [
        ("Vencedor do ciclo", "vencedor"), ("Metrica do desafiante", "metrica_desafiante"),
        ("Metrica do vigente (reavaliado)", "metrica_vigente_reavaliada"), ("Decisao", "decisao"), ("Motivo", "motivo"),
        ("Modelo vigente apos o ciclo", "modelo_vigente_apos_ciclo"), ("Previsoes geradas", "previsoes"),
        ("Previsoes reproduziveis", "previsoes_reproduziveis")]]
    lines += [f"| {name} | " + " | ".join(_fmt(v) for v in values) + " |" for name, values in table]
    out = evidence_root / "comparacao_ciclos.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_pipeline(new_batch: bool, root: Path) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = load_config(root)
    if new_batch:
        try:
            ingestao.release_next_batch(cfg, root)
        except ingestao.BatchesExhausted as exc:
            log.warning("%s. O ciclo roda com os lotes existentes.", exc)

    before = current_cycle_id(cfg, root)
    notebooks_dir = root / "notebooks"
    for name in NOTEBOOKS:
        log.info("Executando %s", name)
        try:
            execute_notebook(notebooks_dir / name)
        except CellExecutionError as exc:
            log.error("Falha em %s: %s: %s. Pipeline interrompido; saidas gravadas no notebook.", name, exc.ename, exc.evalue)
            if (cycle := current_cycle_id(cfg, root)) and cycle != before:
                tag_cycle(load_cycle(cfg, root), cfg, root, f"falhou_em_{name.removesuffix('.ipynb')}")
            return 1

    ctx = load_cycle(cfg, root)
    evidence = ctx.evidence_dir(cfg, root)
    executed_dir = evidence / "notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    for name in NOTEBOOKS:
        shutil.copy2(notebooks_dir / name, executed_dir / name)
    write_json(evidence / "resumo_ciclo.json", cycle_summary(evidence))
    tag_cycle(ctx, cfg, root, "completo")
    log.info("Ciclo %s concluido. Comparacao: %s", ctx.cycle_id, compare_cycles(cfg, root).relative_to(root))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o ciclo MLOps completo (notebooks 01 a 05)")
    parser.add_argument("--novo-lote", action="store_true", help="publica o proximo lote antes do ciclo")
    parser.add_argument("--comparar", action="store_true", help="apenas regenera a comparacao entre ciclos")
    parser.add_argument("--mlflow-ui", action="store_true", help="abre a interface do MLflow sobre mlruns/")
    args = parser.parse_args()
    root = project_root()
    if args.mlflow_ui:
        import mlflow

        configure_mlflow(load_config(root), root)
        sys.exit(subprocess.call([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", mlflow.get_tracking_uri()]))
    if args.comparar:
        print(compare_cycles(load_config(root), root))
        return
    sys.exit(run_pipeline(args.novo_lote, root))


if __name__ == "__main__":
    main()
