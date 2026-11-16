# Previsão de Inadimplência em Cartão de Crédito · Pipeline MLOps

Trabalho Avaliativo N3 · Análise Preditiva.

Este projeto prevê se um cliente de cartão de crédito vai deixar de pagar a fatura do mês seguinte. O modelo é retreinado de forma recorrente: cada novo lote de dados dispara um ciclo completo de validação, seleção de atributos por votação, competição entre dois modelos e decisão automática de promoção. Tudo fica registrado no MLflow.

> **Status.** As fases 1 (bootstrap) e 2 (ingestão + ETL) estão implementadas e verificadas localmente. As seções marcadas como *planejado* descrevem o desenho das fases seguintes e ainda não existem no código.

## Sumário

- [Dataset](#dataset)
- [Arquitetura](#arquitetura)
- [Instalação local](#instalação-local)
- [Execução local](#execução-local)
- [Testes](#testes)
- [Configuração](#configuração)
- [Versionamento de dados e novos lotes](#versionamento-de-dados-e-novos-lotes)
- [Protocolo de avaliação](#protocolo-de-avaliação)
- [Votação de atributos](#votação-de-atributos)
- [Promoção de modelo](#promoção-de-modelo)
- [MLflow](#mlflow)
- [GitHub Actions e persistência](#github-actions-e-persistência)
- [Monitoramento de drift (Critério 10)](#monitoramento-de-drift-critério-10)
- [Reprodutibilidade](#reprodutibilidade)

## Dataset

**Default of Credit Card Clients**, UCI Machine Learning Repository, id 350.

- Fonte: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Licença: Creative Commons Attribution 4.0 International (CC BY 4.0), conforme a página do dataset na UCI.
- DOI: 10.24432/C55S3H
- Citação: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- Contexto: clientes de um banco de Taiwan, abril a setembro de 2005. Valores em dólar taiwanês (NT$).

Números abaixo medidos no arquivo original (`default of credit card clients.xls`, SHA-256 `30c6be3a…`):

| Item | Valor |
|---|---|
| Observações | 30.000 |
| Colunas | 25 (ID, 23 atributos, 1 alvo), todas inteiras |
| Alvo | `default payment next month`, renomeado para `default.payment.next.month` |
| Classe positiva (inadimplente) | 22,12% (desbalanceado) |
| Valores ausentes | 0 |
| IDs duplicados | 0 |
| Linhas idênticas ignorando o ID | 35. São clientes distintos com o mesmo perfil e foram mantidas |
| Códigos fora do dicionário | `EDUCATION` ∈ {0, 5, 6} (345 linhas) e `MARRIAGE` = 0 (54 linhas). Recodificados para "outros" |
| `PAY_x` = -2 e 0 | Não documentados, mas frequentes. Aceitos como "sem consumo" e "crédito rotativo" |
| Outliers (IQR) | De 8% a 10% nas colunas de fatura e pagamento. São valores reais e foram mantidos |

Principais atributos:

| Variável | Significado |
|---|---|
| `LIMIT_BAL` | Limite de crédito (titular + família) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Dados demográficos |
| `PAY_0`, `PAY_2`..`PAY_6` | Status de pagamento de set/2005 a abr/2005. -1 = em dia, 1..9 = meses de atraso. A fonte não tem `PAY_1` |
| `BILL_AMT1..6` | Valor da fatura de set/2005 a abr/2005. Negativo = saldo credor |
| `PAY_AMT1..6` | Valor pago de set/2005 a abr/2005 |

**Limitações.** Os dados vêm de um único banco, em um único país, em 2005, logo de um período e de uma cultura de crédito específicos. A fonte não traz renda nem histórico além de 6 meses. As variáveis `SEX`, `AGE` e `MARRIAGE` são sensíveis do ponto de vista de equidade, e um uso real exigiria uma análise de viés. Os "lotes" de chegada são uma simulação: a fonte é estática e foi particionada.

## Arquitetura

```text
GitHub Actions (schedule + workflow_dispatch)          [planejado: fase 9]
  │
  ├─ 01_etl.ipynb               ingestão → versionamento → validação → limpeza
  │                             → engenharia de atributos → partição      [implementado]
  ├─ 02_votacao_features.ipynb  SelectKBest + RF importance + RFE → votação  [fase 3]
  ├─ 03_treino_competicao.ipynb LogisticRegression vs RandomForest          [fase 4]
  ├─ 04_selecao_promocao.ipynb  vencedor vs modelo vigente → promoção       [fase 5]
  ├─ 05_previsoes.ipynb         previsões com o modelo vigente              [fase 7]
  └─ MLflow                     registro de todo o ciclo                    [fase 6]
```

Os notebooks orquestram e documentam. A lógica reutilizável fica em `src/`:

| Módulo | Responsabilidade |
|---|---|
| `src/utils.py` | Raiz do projeto, configuração, logging, hashing e contexto do ciclo (`cycle_id`, `data_version`) |
| `src/ingestao.py` | Download verificado, partição em lotes, publicação de lote novo, `data_version` |
| `src/preprocessamento.py` | Validação por esquema, limpeza, outliers, atributos derivados, partição e `ColumnTransformer` |
| `src/features.py`, `src/modelagem.py`, `src/promocao.py` | Fases 3 a 5 (*planejado*) |

`cycle_id` e `data_version` nascem no notebook 01. Ficam gravados em `reports/ciclo_atual.json` e os notebooks seguintes leem esse arquivo com `load_cycle`.

## Instalação local

Requer Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (PowerShell ou cmd)
pip install -r requirements.txt
```

## Execução local

```bash
# 1. Baixa a fonte da UCI (confere SHA-256) e publica o lote inicial
python -m src.ingestao

# 2. Executa o ETL e abre um novo ciclo
jupyter nbconvert --to notebook --execute --inplace notebooks/01_etl.ipynb
```

Para inspecionar de forma interativa, rode `jupyter lab` e execute os notebooks na ordem 01 → 05. O passo 1 é opcional, porque o notebook 01 também garante a fonte e o lote inicial.

## Testes

```bash
pytest
```

Os testes usam dados sintéticos e não precisam de rede. Hoje eles cobrem a validação estrutural, a recodificação e o limite de linhas inválidas, os atributos derivados, a estabilidade da partição quando chegam lotes novos, o ajuste do pré-processador apenas no treino, a partição em lotes e o determinismo da `data_version`.

## Configuração

Todos os valores ajustáveis ficam em `config/config.yaml`:

| Seção | Conteúdo |
|---|---|
| `project` | Nome do projeto e do experimento MLflow |
| `data.source` | URL, nome e SHA-256 do arquivo original, nome original do alvo |
| `data.target`, `data.id_column`, `data.random_state` | Alvo, chave do cliente, semente global (42) |
| `data.paths` | Caminhos relativos à raiz do projeto |
| `data.batches` | Frações da fonte usadas como lotes (0,6 / 0,2 / 0,1) e como conjunto de inferência (0,1) |
| `data.validation.max_invalid_row_fraction` | Fração máxima de linhas inválidas antes de o ETL falhar (1%) |
| `evaluation` | Métrica primária (`f1`), direção (`maximize`), `test_size` e `validation_size` |
| `features` | Votos mínimos e quantos atributos cada método seleciona |
| `models` | Habilita os modelos A e B |
| `promotion` | `enabled` e `minimum_improvement` |

Existe um único arquivo de configuração, em `config/config.yaml`. O esqueleto do enunciado também lista um `config.yaml` na raiz. Ele foi omitido de propósito, porque duas cópias divergiriam com o tempo.

## Versionamento de dados e novos lotes

- O arquivo original da UCI fica em `data/raw/source/`. É baixado uma vez, conferido por SHA-256 e **nunca alterado**. Não vai para o Git.
- `partition_source` embaralha a fonte com `random_state` e a divide em `lote_001` (18.000 linhas), `lote_002` (6.000), `lote_003` (3.000) e `data/raw/inferencia/novos_clientes.csv` (3.000 linhas, sem o alvo). Esse último arquivo é a base de previsões "nunca vistas".
- `data/raw/batches/` contém só os lotes já **publicados**. Para simular a chegada de dados novos:

  ```bash
  python -m src.ingestao --novo-lote
  ```

- `data_version = "dv_" + sha256(nome:sha256(conteúdo) de cada lote publicado)[:16]`. Os mesmos lotes sempre geram a mesma versão, e qualquer lote novo ou alterado gera outra. Com só o `lote_001` publicado, a versão é `dv_a2d3d2d25ce5c5c8`.
- `cycle_id = ciclo_<UTC>_<8 primeiros caracteres da data_version>`. No GitHub Actions ele pode ser imposto pela variável `PIPELINE_CYCLE_ID`.

## Protocolo de avaliação

Definido no notebook 01, antes de qualquer treino.

- **Partição por hash do ID.** `sha256("42:<ID>")` mapeia cada cliente para [0, 1]: abaixo de 0,2 vai para **teste**, entre 0,2 e 0,4 para **validação**, acima disso para **treino** (60/20/20).
- **Motivo.** Um `train_test_split` refeito a cada ciclo moveria clientes do teste antigo para o treino novo. Com o hash, cada cliente fica sempre na mesma partição. Isso foi verificado: ao publicar o `lote_002`, os 18.000 clientes do `lote_001` mantiveram a partição.
- **Uso.** O treino ajusta o pré-processamento, a seleção de atributos e os modelos. A validação decide a competição e a promoção. O teste serve só para a avaliação final.
- **Sem vazamento.** O ETL aplica apenas transformações linha a linha. Imputação, one-hot e padronização ficam em `build_preprocessor` (`ColumnTransformer`) e são ajustadas só no treino.
- **Métrica primária.** F1, a ser maximizado (`evaluation.primary_metric`). Com 22% de positivos, um modelo que sempre prevê "adimplente" teria 78% de acurácia e F1 = 0. Métricas secundárias: precisão, recall, acurácia, ROC-AUC e matriz de confusão.

## Votação de atributos

*Planejado (fase 3).* Três métodos independentes rodam no mesmo ciclo, sobre o treino: `SelectKBest`, a importância de uma `RandomForestClassifier` e `RFE`. Cada um registra atributo, ranking ou importância e se o selecionou. Um atributo é eleito quando recebe pelo menos `features.voting.minimum_votes` votos (padrão: 2 de 3). Os dois modelos usam exatamente o conjunto eleito.

## Promoção de modelo

*Planejado (fase 5).* Regra: `f1_novo >= f1_vigente + promotion.minimum_improvement`. Se não houver modelo vigente, o primeiro vencedor é promovido. Versões anteriores nunca são sobrescritas. `models/modelo_vigente.json` aponta para a versão ativa.

## MLflow

*Planejado (fase 6).* Cada run registra `cycle_id`, `data_version`, modelo, hiperparâmetros, sementes, atributos eleitos, métricas e o artefato do modelo. A interface local será aberta com `mlflow ui`. O comando exato será documentado com a fase.

## GitHub Actions e persistência

*Planejado (fases 9 e 11).* Um único workflow, `.github/workflows/pipeline_mlops.yml`, com `schedule` e `workflow_dispatch`, executa os notebooks 01 a 05 em sequência e para no primeiro erro. Os runners são efêmeros, então o estado (lotes publicados, histórico do MLflow, modelos e manifesto) precisa ser persistido explicitamente. A estratégia será documentada aqui na fase 9.

## Monitoramento de drift (Critério 10)

*Planejado (fase 10).* Comparação estatística entre a distribuição de treino de referência e cada lote novo, com relatório, registro no MLflow e um alerta ou decisão no pipeline.

## Reprodutibilidade

```bash
git clone <url-do-repositorio>
cd mlops-credit-default-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_etl.ipynb
```

Pontos que garantem o mesmo resultado em outra máquina:

- O checksum da fonte é conferido.
- A semente é fixa (42) para a partição em lotes e para a partição treino/validação/teste.
- A `data_version` é derivada do conteúdo dos dados.
- As dependências têm versão fixa.
- Todos os caminhos são relativos à raiz do projeto.
