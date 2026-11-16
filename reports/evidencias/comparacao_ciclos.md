# Comparacao entre ciclos

Gerado por `python -m src.pipeline`. Fonte: `reports/evidencias/<cycle_id>/`.

| Item | Ciclo 1 | Ciclo 2 |
|---|---|---|
| cycle_id | ciclo_20261116T223751Z_a2d3d2d2 | ciclo_20261116T223819Z_2db70707 |
| data_version | dv_a2d3d2d25ce5c5c8 | dv_2db70707140b1359 |
| Lotes | lote_001.csv | lote_001.csv, lote_002.csv |
| Linhas | 18000 | 24000 |
| Drift (PSI) | sem_referencia | estavel |
| Atributos eleitos | LIMIT_BAL, PAY_0, PAY_2, PAY_3, PAY_4, PAY_AMT1, UTILIZACAO_MEDIA, MESES_EM_ATRASO, ATRASO_MAXIMO | LIMIT_BAL, PAY_0, PAY_2, PAY_3, PAY_6, PAY_AMT1, UTILIZACAO_MEDIA, MESES_EM_ATRASO, ATRASO_MAXIMO |
| modelo_A val_f1 | 0.5293 | 0.5295 |
| modelo_A val_roc_auc | 0.7530 | 0.7500 |
| modelo_B val_f1 | 0.5439 | 0.5410 |
| modelo_B val_roc_auc | 0.7829 | 0.7783 |
| Vencedor do ciclo | modelo_B | modelo_B |
| Metrica do desafiante | 0.5439 | 0.5410 |
| Metrica do vigente (reavaliado) | - | 0.5403 |
| Decisao | promovido_inicial | mantido |
| Motivo | nao existe modelo vigente: o vencedor do ciclo vira o modelo inicial | reprovado: f1_desafiante 0.5410 vs f1_vigente 0.5403 + 0.0100 |
| Modelo vigente apos o ciclo | ciclo_20261116T223751Z_a2d3d2d2__random_forest | ciclo_20261116T223751Z_a2d3d2d2__random_forest |
| Previsoes geradas | 3000 | 3000 |
| Previsoes reproduziveis | True | True |
