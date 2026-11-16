"""ETL, validacao de dados, engenharia de atributos e particionamento treino/validacao/teste.

Tudo aqui e transformacao linha a linha ou analise descritiva: nada e "ajustado" aos dados.
Transformacoes que aprendem parametros (imputacao, one-hot, padronizacao) ficam em
`build_preprocessor`, que devolve um `ColumnTransformer` a ser ajustado so no treino.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import CycleContext, get_logger, resolve_path, write_json

log = get_logger(__name__)

PAY_STATUS = [f"PAY_{m}" for m in (0, 2, 3, 4, 5, 6)]
BILL_AMT = [f"BILL_AMT{m}" for m in range(1, 7)]
PAY_AMT = [f"PAY_AMT{m}" for m in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


@dataclass(frozen=True)
class Rule:
    """Dominio valido de uma coluna: conjunto fechado de codigos ou intervalo [min, max]."""

    allowed: frozenset[int] | None = None
    min: float | None = None
    max: float | None = None

    def invalid_mask(self, s: pd.Series) -> pd.Series:
        bad = pd.Series(False, index=s.index)
        if self.allowed is not None:
            bad |= ~s.isin(self.allowed)
        if self.min is not None:
            bad |= s < self.min
        if self.max is not None:
            bad |= s > self.max
        return bad & s.notna()


# Dicionario da UCI: EDUCATION 1..4, MARRIAGE 1..3, PAY_x -1..9. O valor -2 ("sem consumo")
# e 0 ("credito rotativo") aparecem nos dados reais e sao aceitos. BILL_AMT negativo
# representa saldo credor do cliente e tambem e valido.
SCHEMA: dict[str, Rule] = {
    "ID": Rule(min=1),
    "LIMIT_BAL": Rule(min=1),
    "SEX": Rule(allowed=frozenset({1, 2})),
    "EDUCATION": Rule(allowed=frozenset({1, 2, 3, 4})),
    "MARRIAGE": Rule(allowed=frozenset({1, 2, 3})),
    "AGE": Rule(min=18, max=100),
    **{c: Rule(min=-2, max=9) for c in PAY_STATUS},
    **{c: Rule() for c in BILL_AMT},
    **{c: Rule(min=0) for c in PAY_AMT},
}
TARGET_RULE = Rule(allowed=frozenset({0, 1}))

# Codigos fora do dicionario oficial, mas presentes na fonte, mapeados para "outros".
UNDOCUMENTED_CODES: dict[str, dict[int, int]] = {
    "EDUCATION": {0: 4, 5: 4, 6: 4},
    "MARRIAGE": {0: 3},
}

OUTLIER_COLUMNS = ["LIMIT_BAL", "AGE", *BILL_AMT, *PAY_AMT]


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_columns: list[str]
    unexpected_columns: list[str]
    missing_values: dict[str, int]
    duplicate_ids: int
    duplicate_rows_excluding_id: int
    invalid_values: dict[str, int]
    undocumented_codes: dict[str, dict[int, int]]
    class_balance: dict[int, float]
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rename_target(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Padroniza o nome do alvo para o definido em `data.target`."""
    return df.rename(columns={cfg["data"]["source"]["original_target"]: cfg["data"]["target"]})


def validate(df: pd.DataFrame, cfg: dict[str, Any]) -> ValidationReport:
    """Valida estrutura, nulos, duplicatas e dominios. Nao altera o DataFrame."""
    target = cfg["data"]["target"]
    id_col = cfg["data"]["id_column"]
    expected = [*SCHEMA, target]
    missing_cols = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected and c != "lote"]
    present = [c for c in SCHEMA if c in df.columns]

    errors: list[str] = []
    if missing_cols:
        errors.append(f"colunas ausentes: {missing_cols}")
    non_numeric = [c for c in [*present, target] if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"colunas nao numericas: {non_numeric}")
        present = [c for c in present if c not in non_numeric]

    invalid = {c: int(SCHEMA[c].invalid_mask(df[c]).sum()) for c in present}
    undocumented = {
        c: {int(k): int(v) for k, v in df[c].value_counts().items() if k in codes}
        for c, codes in UNDOCUMENTED_CODES.items()
        if c in present
    }

    dup_ids = int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
    if dup_ids:
        errors.append(f"{dup_ids} IDs duplicados")

    balance: dict[int, float] = {}
    if target in df.columns and target not in non_numeric:
        if df[target].isna().any():
            errors.append(f"{int(df[target].isna().sum())} linhas sem rotulo")
        if TARGET_RULE.invalid_mask(df[target]).any():
            errors.append("alvo com valores fora de {0, 1}")
        balance = {int(k): round(float(v), 4) for k, v in df[target].value_counts(normalize=True).sort_index().items()}

    feature_cols = [c for c in df.columns if c not in (id_col, "lote")]
    return ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_columns=missing_cols,
        unexpected_columns=unexpected,
        missing_values={c: int(n) for c, n in df.isna().sum().items() if n},
        duplicate_ids=dup_ids,
        duplicate_rows_excluding_id=int(df.duplicated(subset=feature_cols).sum()) if not missing_cols else 0,
        invalid_values={c: n for c, n in invalid.items() if n},
        undocumented_codes={c: v for c, v in undocumented.items() if v},
        class_balance=balance,
        errors=errors,
    )


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recodifica codigos nao documentados e remove linhas ainda invalidas.

    Falha se a fracao removida passar de `data.validation.max_invalid_row_fraction`.
    """
    out = df.copy()
    recoded: dict[str, int] = {}
    for col, mapping in UNDOCUMENTED_CODES.items():
        hits = out[col].isin(mapping)
        recoded[col] = int(hits.sum())
        out[col] = out[col].replace(mapping)

    target = cfg["data"]["target"]
    bad = pd.Series(False, index=out.index)
    for col, rule in SCHEMA.items():
        bad |= rule.invalid_mask(out[col])
    bad |= out[target].isna() | TARGET_RULE.invalid_mask(out[target])

    limit = cfg["data"]["validation"]["max_invalid_row_fraction"]
    if bad.mean() > limit:
        raise DataValidationError(f"{bad.mean():.2%} das linhas invalidas excede o limite de {limit:.2%}")
    out = out.loc[~bad].reset_index(drop=True)
    out[target] = out[target].astype(int)
    return out, {**{f"recodificados_{k}": v for k, v in recoded.items()}, "linhas_removidas": int(bad.sum())}


def outlier_summary(df: pd.DataFrame, columns: list[str] = OUTLIER_COLUMNS) -> pd.DataFrame:
    """Conta valores fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] por coluna. Apenas diagnostico."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"coluna": col, "q1": q1, "q3": q3, "limite_inf": low, "limite_sup": high,
                     "min": df[col].min(), "max": df[col].max(), "n_outliers": n, "pct_outliers": round(n / len(df), 4)})
    return pd.DataFrame(rows)


ENGINEERED = ["UTILIZACAO_ATUAL", "UTILIZACAO_MEDIA", "RAZAO_PAGAMENTO", "MESES_EM_ATRASO", "ATRASO_MAXIMO", "TENDENCIA_FATURA"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos de comportamento de credito. Cada um usa apenas a propria linha."""
    out = df.copy()
    limit = out["LIMIT_BAL"].astype(float)
    bills = out[BILL_AMT].astype(float)
    paid = out[PAY_AMT].astype(float).sum(axis=1)
    owed = bills.clip(lower=0).sum(axis=1)
    out["UTILIZACAO_ATUAL"] = bills["BILL_AMT1"] / limit
    out["UTILIZACAO_MEDIA"] = bills.mean(axis=1) / limit
    # Sem fatura positiva no semestre, nao ha divida: tratado como pagamento integral.
    out["RAZAO_PAGAMENTO"] = np.where(owed > 0, paid / owed.where(owed > 0, 1), 1.0).clip(0, 5)
    out["MESES_EM_ATRASO"] = (out[PAY_STATUS] >= 1).sum(axis=1)
    out["ATRASO_MAXIMO"] = out[PAY_STATUS].max(axis=1)
    out["TENDENCIA_FATURA"] = (bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit
    return out


def assign_split(ids: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Atribui treino/validacao/teste por hash do ID com semente.

    Cada cliente cai sempre na mesma particao, em qualquer ciclo. Assim lotes novos nunca
    movem registros do teste para o treino e o teste permanece isolado entre ciclos.
    """
    seed = cfg["data"]["random_state"]
    test = cfg["evaluation"]["test_size"]
    val = cfg["evaluation"]["validation_size"]
    buckets = ids.map(lambda i: int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return pd.Series(np.select([buckets < test, buckets < test + val], ["teste", "validacao"], "treino"), index=ids.index)


def feature_columns(cfg: dict[str, Any], df: pd.DataFrame) -> list[str]:
    """Colunas candidatas a atributo: tudo exceto ID, alvo e metadados de linhagem."""
    exclude = {cfg["data"]["id_column"], cfg["data"]["target"], "lote", "particao"}
    return [c for c in df.columns if c not in exclude]


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Pre-processador com parametros aprendidos. Deve ser ajustado apenas no treino."""
    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        verbose_feature_names_out=False,
    )


def save_processed(df: pd.DataFrame, ctx: CycleContext, cfg: dict[str, Any], report: dict[str, Any],
                   root: Path | None = None) -> Path:
    """Grava o dataset processado e os metadados do ETL vinculados a cycle_id/data_version."""
    out_dir = resolve_path(cfg, "processed_dir", root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dataset_processado.parquet"
    df.to_parquet(path, index=False)
    meta = {"cycle_id": ctx.cycle_id, "data_version": ctx.data_version, "arquivo": path.name,
            "n_linhas": len(df), "particoes": df["particao"].value_counts().to_dict(), **report}
    write_json(out_dir / "etl_metadata.json", meta)
    write_json(ctx.evidence_dir(cfg, root) / "01_etl.json", meta)
    log.info("Dataset processado salvo: %s (%d linhas, %s)", path.name, len(df), ctx.data_version)
    return path
