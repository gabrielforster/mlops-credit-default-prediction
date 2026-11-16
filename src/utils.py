"""Utilitarios compartilhados: raiz do projeto, configuracao, logging, hashing e contexto de ciclo."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
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
    """Converte um caminho relativo de `data.paths` em caminho absoluto a partir da raiz."""
    return (root or project_root()) / cfg["data"]["paths"][key]


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


@dataclass(frozen=True)
class CycleContext:
    """Identidade de um ciclo do pipeline, propagada de notebook em notebook."""

    cycle_id: str
    data_version: str
    started_at: str
    batches: list[str]

    def evidence_dir(self, cfg: dict[str, Any], root: Path | None = None) -> Path:
        return resolve_path(cfg, "evidence_dir", root) / self.cycle_id


def start_cycle(cfg: dict[str, Any], data_version: str, batches: list[str], root: Path | None = None) -> CycleContext:
    """Abre um novo ciclo e grava o ponteiro lido pelos notebooks seguintes.

    `PIPELINE_CYCLE_ID` permite que o GitHub Actions imponha o id (ex.: numero do run).
    """
    now = datetime.now(timezone.utc)
    cycle_id = os.environ.get("PIPELINE_CYCLE_ID") or f"ciclo_{now:%Y%m%dT%H%M%SZ}_{data_version.removeprefix('dv_')[:8]}"
    ctx = CycleContext(cycle_id=cycle_id, data_version=data_version, started_at=now.isoformat(), batches=batches)
    write_json(resolve_path(cfg, "cycle_pointer", root), asdict(ctx))
    return ctx


def load_cycle(cfg: dict[str, Any], root: Path | None = None) -> CycleContext:
    pointer = resolve_path(cfg, "cycle_pointer", root)
    if not pointer.is_file():
        raise FileNotFoundError(f"Nenhum ciclo ativo em {pointer}. Execute o notebook 01_etl primeiro.")
    return CycleContext(**read_json(pointer))
