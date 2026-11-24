"""Demonstracao de uso: uma entrada nova produz uma resposta compreensivel.

Usa o modelo vigente (`models/modelo_vigente.json`) para avaliar um cliente e explica a resposta:
- pontuacao de risco e decisao (inadimplente ou nao, no limiar do proprio modelo);
- fatores que mais pesaram: para cada atributo usado pelo modelo, quanto a pontuacao muda se o
  valor do cliente for trocado pelo valor tipico (mediana do treino). Explicacao local e agnostica
  ao algoritmo, sem dependencias extras.

Uso:
    python -m src.demo --id 11                  # cliente do arquivo de inferencia (nunca visto no treino)
    python -m src.demo --arquivo cliente.json   # cliente informado em JSON (23 atributos da UCI)
    python -m src.demo --modelo-json            # imprime um JSON de exemplo para editar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src import ingestao, preprocessamento as pp
from src.promocao import load_active_model
from src.utils import load_config, project_root, resolve_path

RAW_COLUMNS = list(pp.SCHEMA)

# Descricao curta dos atributos que o modelo pode usar, para a explicacao em linguagem simples.
DESCRICOES = {
    "LIMIT_BAL": "limite de credito (NT$)",
    "PAY_0": "status de pagamento em set/2005 (-1 em dia, n = meses de atraso)",
    "PAY_2": "status de pagamento em ago/2005",
    "PAY_3": "status de pagamento em jul/2005",
    "PAY_4": "status de pagamento em jun/2005",
    "PAY_5": "status de pagamento em mai/2005",
    "PAY_6": "status de pagamento em abr/2005",
    "PAY_AMT1": "valor pago em set/2005 (NT$)",
    "UTILIZACAO_ATUAL": "fatura de set/2005 / limite",
    "UTILIZACAO_MEDIA": "fatura media do semestre / limite",
    "RAZAO_PAGAMENTO": "total pago / total faturado no semestre",
    "MESES_EM_ATRASO": "meses com atraso no semestre",
    "ATRASO_MAXIMO": "maior atraso no semestre (meses)",
}

EXEMPLO = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 45000,
    "BILL_AMT4": 30000, "BILL_AMT5": 28000, "BILL_AMT6": 27000,
    "PAY_AMT1": 0, "PAY_AMT2": 1500, "PAY_AMT3": 1200, "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
}


def faixa_de_risco(score: float) -> str:
    if score < 0.3:
        return "baixo"
    if score < 0.5:
        return "moderado"
    if score < 0.7:
        return "alto"
    return "muito alto"


def valores_tipicos(manifest: dict[str, Any], features: list[str], cfg: dict[str, Any], root: Path) -> pd.Series:
    """Mediana de cada atributo no treino do modelo vigente (lotes do manifesto, particao treino)."""
    lotes = {Path(b).stem for b in manifest["batches"]}
    dataset = pp.build_dataset(ingestao.load_released_batches(cfg, root), cfg)
    treino = dataset[(dataset["particao"] == "treino") & dataset["lote"].isin(lotes)]
    return treino[features].median()


def avaliar_cliente(cliente: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Avalia um cliente (atributos brutos da UCI) com o modelo vigente e explica a pontuacao."""
    root = root or project_root()
    cfg = load_config(root)
    active = load_active_model(cfg, root)
    if active is None:
        raise RuntimeError("Nenhum modelo vigente. Rode o pipeline antes (python -m src.pipeline).")
    bundle, manifest = active

    faltando = [c for c in RAW_COLUMNS if c != "ID" and c not in cliente]
    if faltando:
        raise ValueError(f"Atributos ausentes: {faltando}")
    raw = pd.DataFrame([{"ID": 1, **{c: cliente[c] for c in RAW_COLUMNS if c != "ID"}}])
    linha, log = pp.prepare_inference(raw, cfg)
    if linha.empty:
        invalidos = [c for c, rule in pp.SCHEMA.items() if bool(rule.invalid_mask(raw[c]).iloc[0])]
        raise ValueError(f"Valores fora do dominio do dicionario da UCI: {invalidos}")

    features = bundle["features"]
    model = bundle["pipeline"]
    X = linha[features]
    score = float(model.predict_proba(X)[:, 1][0])
    classe = int(model.predict(X)[0])

    tipicos = valores_tipicos(manifest, features, cfg, root)
    variantes = pd.concat([X] * len(features), ignore_index=True)
    for i, f in enumerate(features):
        variantes.loc[i, f] = tipicos[f]
    sem_cada = model.predict_proba(variantes)[:, 1]
    fatores = sorted(
        ({"atributo": f, "descricao": DESCRICOES.get(f, f), "valor_cliente": float(X.iloc[0][f]),
          "valor_tipico": float(tipicos[f]), "efeito": score - float(s)} for f, s in zip(features, sem_cada)),
        key=lambda d: abs(d["efeito"]), reverse=True)

    return {"pontuacao": score, "classe": classe, "faixa": faixa_de_risco(score), "fatores": fatores,
            "modelo": manifest["model_id"], "algoritmo": manifest["algorithm"], "atributos": features,
            "recodificados": {k: v for k, v in log.items() if v}}


def _num(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".") if abs(v) >= 1000 else f"{v:.2f}".rstrip("0").rstrip(".")


def explicar(r: dict[str, Any], top: int = 5) -> str:
    decisao = "PROVAVEL INADIMPLENTE" if r["classe"] == 1 else "provavel adimplente"
    linhas = [
        f"Modelo vigente : {r['modelo']} ({r['algoritmo']})",
        f"Pontuacao      : {r['pontuacao']:.1%}  (risco {r['faixa']})",
        f"Decisao        : {decisao}  (limiar 50%)",
        "",
        "O que mais pesou (pontos percentuais que o valor do cliente soma ou tira da pontuacao,",
        "comparado ao valor tipico do treino):",
    ]
    for f in r["fatores"][:top]:
        linhas.append(f"  {f['efeito'] * 100:+6.1f} p.p.  {f['atributo']:<17} cliente={_num(f['valor_cliente']):>8}"
                      f"  tipico={_num(f['valor_tipico']):>8}  ({f['descricao']})")
    linhas += ["", "A pontuacao ordena clientes por risco. O modelo usa class_weight='balanced', entao ela fica",
               "acima da frequencia real de inadimplencia (22%) e nao deve ser lida como probabilidade calibrada."]
    return "\n".join(linhas)


def cliente_por_id(cliente_id: int, root: Path) -> dict[str, Any]:
    cfg = load_config(root)
    df = pd.read_csv(resolve_path(cfg, "inference_file", root))
    row = df[df["ID"] == cliente_id]
    if row.empty:
        raise SystemExit(f"ID {cliente_id} nao esta em {cfg['paths']['inference_file']}. Exemplos: {df['ID'].head(5).tolist()}")
    return row.iloc[0].to_dict()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Avalia o risco de inadimplencia de um cliente com o modelo vigente")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--id", type=int, help="ID de um cliente do arquivo de inferencia (sem rotulo)")
    grupo.add_argument("--arquivo", type=Path, help="JSON com os 23 atributos brutos do cliente")
    grupo.add_argument("--modelo-json", action="store_true", help="imprime um JSON de exemplo e sai")
    parser.add_argument("--json", action="store_true", help="saida em JSON em vez de texto")
    args = parser.parse_args(argv)

    if args.modelo_json:
        print(json.dumps(EXEMPLO, indent=2))
        return
    root = project_root()
    cliente = cliente_por_id(args.id, root) if args.id is not None else json.loads(args.arquivo.read_text(encoding="utf-8"))
    try:
        resultado = avaliar_cliente(cliente, root)
    except ValueError as exc:
        raise SystemExit(f"Entrada invalida: {exc}") from exc
    print(json.dumps(resultado, indent=2, ensure_ascii=False) if args.json else explicar(resultado))


if __name__ == "__main__":
    main(sys.argv[1:])
