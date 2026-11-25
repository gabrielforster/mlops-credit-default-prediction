"""Demonstracao visual: pagina web local para avaliar clientes com o modelo vigente.

So biblioteca padrao (http.server) e uma pagina HTML sem dependencias externas, para rodar
sem internet durante a apresentacao. A logica de avaliacao e a mesma de `src/demo.py`.

Uso:
    python -m src.demo_web               # abre http://127.0.0.1:8000 no navegador
    python -m src.demo_web --porta 8080 --sem-navegador
"""
from __future__ import annotations

import argparse
import json
import math
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from src.demo import EXEMPLO, RAW_COLUMNS, avaliar_cliente
from src.promocao import load_manifest
from src.utils import load_config, project_root, resolve_path

PAGE = Path(__file__).with_name("web") / "demo.html"

# Cliente que paga a fatura inteira todo mes: contraponto ao EXEMPLO de alto risco.
EM_DIA = {**EXEMPLO, "LIMIT_BAL": 200000, "AGE": 38,
          **{f"PAY_{m}": -1 for m in (0, 2, 3, 4, 5, 6)},
          **{f"BILL_AMT{m}": v for m, v in zip(range(1, 7), (12000, 9500, 11000, 8000, 10500, 9000))},
          **{f"PAY_AMT{m}": v for m, v in zip(range(1, 7), (9500, 11000, 8000, 10500, 9000, 9800))}}

PRESETS = {
    "alto_risco": {"rotulo": "Atrasou 2 meses, limite quase todo usado", "cliente": EXEMPLO},
    "em_dia": {"rotulo": "Paga a fatura inteira todo mes", "cliente": EM_DIA},
}


@lru_cache(maxsize=1)
def clientes_inferencia(root: Path) -> pd.DataFrame:
    """Os 3.000 clientes sem rotulo, nunca vistos no treino."""
    return pd.read_csv(resolve_path(load_config(root), "inference_file", root)).set_index("ID")


def info(root: Path) -> dict[str, Any]:
    manifest = load_manifest(load_config(root), root) or {}
    return {"modelo": manifest.get("model_id"), "algoritmo": manifest.get("algorithm"),
            "atributos": manifest.get("selected_features", []), "campos": [c for c in RAW_COLUMNS if c != "ID"],
            "presets": PRESETS, "ids": clientes_inferencia(root).index[:40].tolist()}


def cliente(root: Path, cliente_id: int) -> dict[str, Any]:
    df = clientes_inferencia(root)
    if cliente_id not in df.index:
        raise KeyError(cliente_id)
    return {k: int(v) for k, v in df.loc[cliente_id].items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(_json_safe(payload), ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (nome exigido pelo http.server)
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(info(root))
            elif path.startswith("/api/cliente/"):
                try:
                    self._json(cliente(root, int(path.rsplit("/", 1)[1])))
                except (KeyError, ValueError):
                    self._json({"erro": "ID nao encontrado entre os clientes de inferencia"}, HTTPStatus.NOT_FOUND)
            else:
                self._json({"erro": "rota inexistente"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/avaliar":
                self._json({"erro": "rota inexistente"}, HTTPStatus.NOT_FOUND)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                dados = {k: float(v) for k, v in body.items() if k in RAW_COLUMNS and v is not None}
                self._json(avaliar_cliente(dados, root))
            except (ValueError, TypeError) as exc:
                self._json({"erro": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:  # silencia o log por requisicao
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demonstracao web do modelo de inadimplencia")
    parser.add_argument("--porta", type=int, default=8000)
    parser.add_argument("--sem-navegador", action="store_true", help="nao abre o navegador automaticamente")
    args = parser.parse_args(argv)

    root = project_root()
    server = ThreadingHTTPServer(("127.0.0.1", args.porta), make_handler(root))
    url = f"http://127.0.0.1:{args.porta}"
    print(f"Demonstracao em {url}  (Ctrl+C para encerrar)")
    if not args.sem_navegador:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
