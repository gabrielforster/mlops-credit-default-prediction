"""Utilitarios compartilhados: raiz do projeto, configuracao, logging, hashing, contexto de ciclo e MLflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CONFIG_RELATIVE_PATH = Path("config") / "config.yaml"


def project_root(start: Path | None = None) -> Path:
    """Sobe a arvore de diretorios ate achar `config/config.yaml`."""
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(f"{CONFIG_RELATIVE_PATH} nao encontrado a partir de {origin}")


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    with open(root / CONFIG_RELATIVE_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(cfg: dict[str, Any], key: str, root: Path | None = None) -> Path:
    """Converte um caminho relativo de `paths` em caminho absoluto a partir da raiz."""
    return (root or project_root()) / cfg["paths"][key]


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- MLflow

def _relocate_artifact_root(db_path: Path, old: str, new: str) -> None:
    """Reescreve o prefixo absoluto dos artefatos gravado no banco do MLflow.

    O MLflow guarda caminhos absolutos. Como `mlruns/` e versionado no Git e o runner do
    GitHub Actions clona o repositorio em outro diretorio, o prefixo precisa acompanhar o clone.
    """
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in [("experiments", "artifact_location"), ("runs", "artifact_uri"),
                              ("logged_models", "artifact_location")]:
            if table in tables:
                conn.execute(f"UPDATE {table} SET {column} = ? || substr({column}, ?) WHERE {column} LIKE ? || '%'",
                             (new, len(old) + 1, old))


def configure_mlflow(cfg: dict[str, Any], root: Path | None = None) -> str:
    """Aponta o MLflow para `mlruns/` na raiz do projeto e devolve o id do experimento."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    root = root or project_root()
    mlruns = resolve_path(cfg, "mlruns_dir", root)
    mlruns.mkdir(parents=True, exist_ok=True)
    db_path = mlruns / "mlflow.db"
    artifact_root = (mlruns / "artifacts").resolve().as_uri()
    name = cfg["project"]["mlflow_experiment"]

    if db_path.is_file():
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT artifact_location FROM experiments WHERE name = ?", (name,)).fetchone() \
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='experiments'").fetchone() else None
        if row and row[0] != artifact_root:
            _relocate_artifact_root(db_path, row[0], artifact_root)

    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve().as_posix()}")
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


# --------------------------------------------------------------------------- Ciclo

@dataclass(frozen=True)
class CycleContext:
    """Identidade de um ciclo do pipeline, propagada de notebook em notebook."""

    cycle_id: str
    data_version: str
    started_at: str
    batches: list[str]
    parent_run_id: str | None = None

    def evidence_dir(self, cfg: dict[str, Any], root: Path | None = None) -> Path:
        return resolve_path(cfg, "evidence_dir", root) / self.cycle_id

    @property
    def tags(self) -> dict[str, str]:
        return {"cycle_id": self.cycle_id, "data_version": self.data_version}


def start_cycle(cfg: dict[str, Any], data_version: str, batches: list[str], root: Path | None = None) -> CycleContext:
    """Abre um ciclo: gera o `cycle_id`, cria o run pai no MLflow e grava o ponteiro do ciclo.

    `PIPELINE_CYCLE_ID` permite que o GitHub Actions imponha o id (ex.: numero do run).
    """
    import mlflow

    now = datetime.now(timezone.utc)
    cycle_id = os.environ.get("PIPELINE_CYCLE_ID") or f"ciclo_{now:%Y%m%dT%H%M%SZ}_{data_version.removeprefix('dv_')[:8]}"
    ctx = CycleContext(cycle_id=cycle_id, data_version=data_version, started_at=now.isoformat(), batches=batches)
    configure_mlflow(cfg, root)
    tags = {**ctx.tags, "stage": "ciclo", "lotes": ",".join(batches), "pipeline_status": "em_execucao"}
    if run_url := os.environ.get("GITHUB_RUN_ID"):
        tags["github_run_id"] = run_url
    with mlflow.start_run(run_name=cycle_id, tags=tags) as run:
        mlflow.log_params({"cycle_id": cycle_id, "data_version": data_version, "n_lotes": len(batches)})
    ctx = replace(ctx, parent_run_id=run.info.run_id)
    write_json(resolve_path(cfg, "cycle_pointer", root), asdict(ctx))
    return ctx


def load_cycle(cfg: dict[str, Any], root: Path | None = None) -> CycleContext:
    pointer = resolve_path(cfg, "cycle_pointer", root)
    if not pointer.is_file():
        raise FileNotFoundError(f"Nenhum ciclo ativo em {pointer}. Execute o notebook 01_etl primeiro.")
    return CycleContext(**read_json(pointer))


@contextmanager
def cycle_run(ctx: CycleContext, cfg: dict[str, Any], stage: str, run_name: str | None = None,
              root: Path | None = None) -> Iterator[Any]:
    """Abre um run filho do run do ciclo, ja marcado com `cycle_id`, `data_version` e `stage`."""
    import mlflow

    configure_mlflow(cfg, root)
    with mlflow.start_run(run_id=ctx.parent_run_id):
        with mlflow.start_run(run_name=run_name or stage, nested=True, tags={**ctx.tags, "stage": stage}) as child:
            yield child
