#!/usr/bin/env bash
# Monta a pasta de entrega (nao versionada) a partir do estado commitado do repositorio.
# Tudo o que esta em entrega/ e copia de arquivos versionados: a fonte da verdade e o repositorio.
# Uso: bash scripts/montar_entrega.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

if ! git diff --quiet HEAD -- . ':(exclude)entrega'; then
  echo "Ha alteracoes nao commitadas; a entrega deve refletir um commit. Commite antes." >&2
  exit 1
fi

OUT="$ROOT/entrega"
COMMIT="$(git rev-parse --short HEAD)"
REMOTE="$(git remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##')"

rm -rf "$OUT"
mkdir -p "$OUT/evidencias"

# (b) relatorio tecnico
cp reports/relatorio_n3.pdf "$OUT/relatorio_n3.pdf"

# (a) link do repositorio
printf '%s\ncommit da entrega: %s\n' "$REMOTE" "$(git rev-parse HEAD)" > "$OUT/link_repositorio.txt"

# (c) e (e) notebooks, codigo, configuracao, workflow e README: snapshot do commit
git archive --format=zip --prefix="mlops-credit-default-prediction/" -o "$OUT/repositorio_${COMMIT}.zip" HEAD

# (d) evidencias do schedule e historico MLflow
cp -r evidencias/. "$OUT/evidencias/"
cp reports/evidencias/comparacao_ciclos.md "$OUT/evidencias/comparacao_ciclos.md"
cp models/modelo_vigente.json "$OUT/evidencias/modelo_vigente.json"
cp models/historico_promocoes.jsonl "$OUT/evidencias/historico_promocoes.jsonl"

SCHEDULE_LOG="$(ls evidencias/github_actions/*_schedule.log 2>/dev/null | head -1 || true)"
if [ -n "$SCHEDULE_LOG" ]; then SCHEDULE_STATUS="incluido: ${SCHEDULE_LOG#evidencias/}"; else SCHEDULE_STATUS="PENDENTE: ainda nao ha log de execucao por schedule"; fi

cat > "$OUT/LEIA-ME.md" <<EOF
# Entrega · Trabalho N3 · Análise Preditiva

Gerado por \`scripts/montar_entrega.sh\` a partir do commit \`${COMMIT}\`.

| Item exigido (seção 7 do enunciado) | Arquivo |
|---|---|
| (a) Link do repositório Git | \`link_repositorio.txt\` |
| (b) Relatório técnico em PDF | \`relatorio_n3.pdf\` |
| (c) Notebooks, código, configuração e workflow | \`repositorio_${COMMIT}.zip\` (snapshot do commit) |
| (d) Evidências do schedule e histórico MLflow | \`evidencias/\` (logs do GitHub Actions, \`mlflow/runs.md\`, \`mlflow/runs.csv\`, comparação de ciclos, manifesto) e \`mlruns/\` dentro do zip |
| (e) README de reprodução | \`README.md\` dentro do zip |

Log de execução por schedule: ${SCHEDULE_STATUS}.
EOF

echo "Entrega montada em entrega/ (commit ${COMMIT})"
ls -la "$OUT"
