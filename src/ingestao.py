"""Ingestao e versionamento de dados.

Fluxo:
1. `download_source` baixa o arquivo original da UCI uma unica vez e confere o SHA-256.
   O arquivo bruto nunca e alterado.
2. `partition_source` divide a fonte, de forma deterministica (semente fixa), em lotes
   que simulam a chegada de dados ao longo do tempo, mais um conjunto de inferencia
   sem rotulo ("novos clientes").
3. `release_next_batch` publica o proximo lote em `data/raw/batches/`. Os lotes
   publicados definem o estado dos dados de treino.
4. `compute_data_version` gera um hash do conteudo dos lotes publicados: os mesmos
   lotes sempre produzem a mesma `data_version`.

Uso pela linha de comando:
    python -m src.ingestao               # garante fonte + lote inicial
    python -m src.ingestao --novo-lote   # publica o proximo lote (novo ciclo de dados)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, project_root, resolve_path, sha256_file

log = get_logger(__name__)

BATCH_PREFIX = "lote_"


class IngestionError(RuntimeError):
    pass


def source_file(cfg: dict[str, Any], root: Path | None = None) -> Path:
    return resolve_path(cfg, "source_dir", root) / cfg["data"]["source"]["file_name"]


def download_source(cfg: dict[str, Any], root: Path | None = None) -> Path:
    """Baixa e extrai o .xls original, se ainda nao existir, e valida o checksum."""
    src = cfg["data"]["source"]
    target = source_file(cfg, root)
    if not target.is_file():
        log.info("Baixando dataset de %s", src["url"])
        with urllib.request.urlopen(src["url"], timeout=120) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(src["file_name"]))
    digest = sha256_file(target)
    if digest != src["sha256"]:
        raise IngestionError(f"Checksum divergente para {target.name}: {digest} != {src['sha256']}")
    log.info("Fonte verificada: %s (sha256 %s...)", target.name, digest[:12])
    return target


def read_source(cfg: dict[str, Any], root: Path | None = None) -> pd.DataFrame:
    """Le o .xls original. A primeira linha da planilha contem rotulos X1..X23/Y; o cabecalho real esta na segunda."""
    return pd.read_excel(source_file(cfg, root), header=1, engine="calamine")


def partition_source(cfg: dict[str, Any], source: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Particiona a fonte em lotes rotulados e um conjunto de inferencia sem rotulo.

    A permutacao usa `data.random_state`, entao a particao e identica em qualquer maquina.
    """
    spec = cfg["data"]["batches"]
    fractions = list(spec["fractions"])
    if not np.isclose(sum(fractions) + spec["inference_fraction"], 1.0):
        raise IngestionError("data.batches.fractions + inference_fraction precisa somar 1")
    order = np.random.default_rng(cfg["data"]["random_state"]).permutation(len(source))
    cuts = np.round(np.cumsum(fractions) * len(source)).astype(int)
    pieces = np.split(order, cuts)
    batches = [source.iloc[np.sort(idx)].reset_index(drop=True) for idx in pieces[:-1]]
    inference = source.iloc[np.sort(pieces[-1])].drop(columns=[cfg["data"]["source"]["original_target"]])
    return batches, inference.reset_index(drop=True)


def batch_name(index: int) -> str:
    return f"{BATCH_PREFIX}{index + 1:03d}.csv"


def released_batches(cfg: dict[str, Any], root: Path | None = None) -> list[Path]:
    return sorted(resolve_path(cfg, "batches_dir", root).glob(f"{BATCH_PREFIX}*.csv"))


def write_inference_set(cfg: dict[str, Any], inference: pd.DataFrame, root: Path | None = None) -> Path:
    path = resolve_path(cfg, "inference_file", root)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        inference.to_csv(path, index=False)
        log.info("Conjunto de inferencia gravado: %s (%d registros)", path.name, len(inference))
    return path


def release_next_batch(cfg: dict[str, Any], root: Path | None = None) -> Path:
    """Publica o proximo lote ainda nao liberado. Falha se todos ja foram publicados."""
    download_source(cfg, root)
    batches, inference = partition_source(cfg, read_source(cfg, root))
    write_inference_set(cfg, inference, root)
    already = {p.name for p in released_batches(cfg, root)}
    for i, batch in enumerate(batches):
        name = batch_name(i)
        if name not in already:
            path = resolve_path(cfg, "batches_dir", root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            batch.to_csv(path, index=False)
            log.info("Lote publicado: %s (%d registros)", name, len(batch))
            return path
    raise IngestionError(f"Todos os {len(batches)} lotes ja foram publicados")


def ensure_initial_batch(cfg: dict[str, Any], root: Path | None = None) -> list[Path]:
    """Garante que existe ao menos o lote inicial publicado. Idempotente."""
    if not released_batches(cfg, root):
        release_next_batch(cfg, root)
    return released_batches(cfg, root)


def compute_data_version(batch_files: list[Path]) -> str:
    """Hash deterministico do conjunto de lotes: nome + SHA-256 do conteudo de cada arquivo."""
    if not batch_files:
        raise IngestionError("Nenhum lote publicado para versionar")
    digest = hashlib.sha256()
    for path in sorted(batch_files, key=lambda p: p.name):
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return f"dv_{digest.hexdigest()[:16]}"


def load_released_batches(cfg: dict[str, Any], root: Path | None = None) -> pd.DataFrame:
    """Concatena os lotes publicados, anotando a origem de cada linha na coluna `lote`."""
    frames = [pd.read_csv(p).assign(lote=p.stem) for p in released_batches(cfg, root)]
    if not frames:
        raise IngestionError("Nenhum lote publicado. Rode `python -m src.ingestao` primeiro.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestao e versionamento dos lotes de dados")
    parser.add_argument("--novo-lote", action="store_true", help="publica o proximo lote de dados")
    args = parser.parse_args()
    root = project_root()
    cfg = load_config(root)
    if args.novo_lote:
        release_next_batch(cfg, root)
    files = ensure_initial_batch(cfg, root)
    log.info("Lotes publicados: %s | data_version=%s", [p.name for p in files], compute_data_version(files))


if __name__ == "__main__":
    main()
