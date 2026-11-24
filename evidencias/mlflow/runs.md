# Historico do MLflow

Gerado por `python -m src.exportar_mlflow` a partir de `mlruns/mlflow.db`.
Experimento: `credit-default-prediction`. Runs: 24.

## ciclo_20261116T223751Z_a2d3d2d2

- run pai: `9a891110dfe14d90877c1b9c2d9b0e09` (2026-11-16 22:37:54 UTC, FINISHED)
- data_version: `dv_a2d3d2d25ce5c5c8` · lotes: lote_001.csv
- pipeline_status: `completo` · decisao: `promovido_inicial` · vigente: `ciclo_20261116T223751Z_a2d3d2d2__random_forest`

| Etapa | run_id | Metricas |
|---|---|---|
| etl | `230c931bfc29435bb98f6ed973ded747` | linhas_treino=10851, linhas_validacao=3613, linhas_teste=3536 |
| drift | `3c84432145a3423b86abf1f9cbe4f208` | status=sem_referencia |
| votacao_features | `69c177be206844eb8eef02752d18a592` | n_elected=9 |
| modelo_A_logistic_regression | `7086266305fd4fccab2ff833eabe0a80` | val_f1=0.5293, val_precision=0.4756, val_recall=0.5967, val_roc_auc=0.7530 |
| modelo_B_random_forest | `fb83652b405640f9a1d261d8d5a48b00` | val_f1=0.5439, val_precision=0.5082, val_recall=0.5849, val_roc_auc=0.7829 |
| promocao | `52c528a54a8c437d8e5d4dea49aa1794` | active_test_f1=0.5249, active_test_roc_auc=0.7794 |
| previsoes | `decc4439873f414caa13fd90bc20da94` | reproducible=1, taxa_prevista_inadimplencia=0.2770 |

## ciclo_20261116T223819Z_2db70707

- run pai: `61619e42212f478aa03afc8de987c0bb` (2026-11-16 22:38:20 UTC, FINISHED)
- data_version: `dv_2db70707140b1359` · lotes: lote_001.csv,lote_002.csv
- pipeline_status: `completo` · decisao: `mantido` · vigente: `ciclo_20261116T223751Z_a2d3d2d2__random_forest`

| Etapa | run_id | Metricas |
|---|---|---|
| etl | `d07732ee30c749e29ab931c91efee262` | linhas_treino=14475, linhas_validacao=4819, linhas_teste=4706 |
| drift | `8f6bf55b78664c08ae3299e695fd48dc` | status=estavel, maior psi_AGE=0.0033 |
| votacao_features | `c0e077dfd18543b7a81798809c3d1cb5` | n_elected=9 |
| modelo_A_logistic_regression | `bba2f1fd7ed8410ca664c30a1477a8df` | val_f1=0.5295, val_precision=0.4790, val_recall=0.5919, val_roc_auc=0.7500 |
| modelo_B_random_forest | `53d7c3a4614c425fa8332befcd6daa74` | val_f1=0.5410, val_precision=0.4904, val_recall=0.6033, val_roc_auc=0.7783 |
| promocao | `47adb9fd22a6403fadbbcedab9e41c5f` | active_test_f1=0.5363, active_test_roc_auc=0.7847 |
| previsoes | `d9922d78a61f4c89be6501e21fa5b8d7` | reproducible=1, taxa_prevista_inadimplencia=0.2770 |

## ciclo_gha_43986932580_1

- run pai: `9e65b546c384463993f40555eba9761d` (2026-11-17 00:22:35 UTC, FINISHED)
- data_version: `dv_3a043c533b992faa` · lotes: lote_001.csv,lote_002.csv,lote_003.csv
- pipeline_status: `completo` · decisao: `mantido` · vigente: `ciclo_20261116T223751Z_a2d3d2d2__random_forest`
- GitHub Actions run: `43986932580`

| Etapa | run_id | Metricas |
|---|---|---|
| etl | `3548ceed42194137be650426e3d31cde` | linhas_treino=16239, linhas_validacao=5439, linhas_teste=5322 |
| drift | `fd45888b656b478cae8ffe5200d48672` | status=estavel, maior psi_AGE=0.0035 |
| votacao_features | `9385e9aff06c47cc8a7abcd48cadc0ed` | n_elected=10 |
| modelo_A_logistic_regression | `304b8409d5ce47bf83e16b23c97527f4` | val_f1=0.5246, val_precision=0.4717, val_recall=0.5909, val_roc_auc=0.7466 |
| modelo_B_random_forest | `c8ebc5d040024e4d9f2c796572fc2301` | val_f1=0.5343, val_precision=0.4849, val_recall=0.5948, val_roc_auc=0.7738 |
| promocao | `e7baf34540694fd298675b0edba000ef` | active_test_f1=0.5455, active_test_roc_auc=0.7879 |
| previsoes | `90ca68862aab4924852f8a9677833418` | reproducible=1, taxa_prevista_inadimplencia=0.2770 |
