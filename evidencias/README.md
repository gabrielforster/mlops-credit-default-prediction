# Evidências

Índice das evidências da entrega, organizado pelo checklist do enunciado. As evidências geradas a cada ciclo pelo próprio pipeline ficam em `reports/evidencias/`. Esta pasta reúne o que vem de fora do ciclo: os logs do GitHub Actions e a exportação do MLflow.

## Conteúdo

| Arquivo | O que comprova |
|---|---|
| `github_actions/run_43986932580_workflow_dispatch.log` | Log completo de um ciclo no GitHub Actions: instalação, 37 testes, notebooks 01→05, upload das evidências e commit do estado (`79d1b2e`) |
| `github_actions/run_<id>_schedule.log` | **Pendente.** Log da primeira execução disparada pelo `schedule` (segunda-feira, 09:00 UTC) |
| `mlflow/runs.md` | Histórico do MLflow por ciclo: run pai, etapas, `run_id`, métricas e decisão |
| `mlflow/runs.csv` | Os mesmos 24+ runs em tabela, com todas as tags e métricas |

Para regenerar a exportação do MLflow: `python -m src.exportar_mlflow`.
Para baixar o log de um run: `gh run view <run_id> --log > evidencias/github_actions/run_<run_id>_<evento>.log`.

## Checklist do enunciado

| Pergunta | Evidência |
|---|---|
| Um workflow agendado executa todo o pipeline? | `.github/workflows/pipeline_mlops.yml` (gatilho `schedule` semanal); log do run agendado nesta pasta; run manual `43986932580` como execução completa já comprovada |
| Há três métodos, votação e features eleitas? | `reports/evidencias/<cycle_id>/02_votacao_features.csv` (uma linha por método e atributo); notebook `02_votacao_features.ipynb` |
| Os dois modelos competem sob avaliação comparável? | `reports/evidencias/<cycle_id>/03_competicao.json`; runs `modelo_A_logistic_regression` e `modelo_B_random_forest` em `mlflow/runs.md` |
| O MLflow permite recuperar os ciclos e suas decisões? | `mlruns/mlflow.db` + `mlruns/artifacts/`; `mlflow/runs.md`; `python -m src.pipeline --mlflow-ui` |
| Há previsão com o vigente e recurso adicional demonstrado? | `reports/previsoes/`, `reports/evidencias/<cycle_id>/05_previsoes.json`; drift em `01_drift.json` e no controle positivo do notebook 01 |

## Ciclos registrados

| Ciclo | Onde rodou | Lotes | Decisão |
|---|---|---|---|
| `ciclo_20261116T223751Z_a2d3d2d2` | Local | `lote_001` | `promovido_inicial` |
| `ciclo_20261116T223819Z_2db70707` | Local | `lote_001` + `lote_002` | `mantido` |
| `ciclo_gha_43986932580_1` | GitHub Actions (`workflow_dispatch`) | `lote_001` a `lote_003` | `mantido` |
