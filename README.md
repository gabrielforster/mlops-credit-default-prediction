# Previsão de Inadimplência em Cartão de Crédito · Pipeline MLOps

Trabalho Avaliativo N3 · Análise Preditiva.

Este projeto prevê se um cliente de cartão de crédito vai deixar de pagar a fatura do mês seguinte. O modelo não é treinado uma vez só. Um workflow do GitHub Actions, disparado manualmente, roda um **ciclo** completo a cada novo lote de dados:

1. validar os dados;
2. eleger atributos por votação de três métodos;
3. treinar dois algoritmos sob o mesmo protocolo e escolher o vencedor;
4. comparar o vencedor com o modelo em produção;
5. promover o vencedor apenas se ele for melhor pela regra configurada;
6. gerar previsões;
7. registrar tudo no MLflow.

## Sumário

- [Dataset](#dataset)
- [Arquitetura](#arquitetura)
- [Instalação local](#instalação-local)
- [Execução local](#execução-local)
- [MLflow](#mlflow)
- [Testes](#testes)
- [Notebooks](#notebooks)
- [Configuração](#configuração)
- [Protocolo de avaliação](#protocolo-de-avaliação)
- [Votação de atributos](#votação-de-atributos)
- [Competição entre modelos](#competição-entre-modelos)
- [Promoção de modelo](#promoção-de-modelo)
- [Versionamento de dados e novos lotes](#versionamento-de-dados-e-novos-lotes)
- [GitHub Actions e persistência](#github-actions-e-persistência)
- [Monitoramento de drift (Critério 10)](#monitoramento-de-drift-critério-10)
- [Demonstração em dois ciclos](#demonstração-em-dois-ciclos)
- [Reprodutibilidade](#reprodutibilidade)
- [Limitações conhecidas](#limitações-conhecidas)

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

Atributos derivados no ETL. Cada um usa apenas a própria linha, então não há vazamento:

| Atributo | Definição |
|---|---|
| `UTILIZACAO_ATUAL` | `BILL_AMT1 / LIMIT_BAL` |
| `UTILIZACAO_MEDIA` | média(`BILL_AMT1..6`) / `LIMIT_BAL` |
| `RAZAO_PAGAMENTO` | Σ`PAY_AMT` / Σ max(`BILL_AMT`, 0), limitado a [0, 5] |
| `MESES_EM_ATRASO` | nº de meses com `PAY_x` ≥ 1 |
| `ATRASO_MAXIMO` | max(`PAY_x`) |
| `TENDENCIA_FATURA` | (`BILL_AMT1` − `BILL_AMT6`) / `LIMIT_BAL` |

**Limitações dos dados.**
- Os dados vêm de um único banco, em um único país, em 2005, ou seja, de um período e de uma cultura de crédito específicos.
- A fonte não traz renda nem histórico além de 6 meses.
- `SEX`, `AGE` e `MARRIAGE` são variáveis sensíveis do ponto de vista de equidade. Um uso real exigiria uma análise de viés.
- Os "lotes" de chegada são uma simulação: a fonte é estática e foi particionada.

## Arquitetura

```text
GitHub Actions: .github/workflows/pipeline_mlops.yml  (workflow_dispatch)
  │  checkout do estado persistido → pip install → pytest
  │
  ├─ python -m src.ingestao --novo-lote   ingestão: publica o próximo lote, data_version = hash do conteúdo
  │
  └─ python -m src.pipeline               executa os notebooks em sequência e para no primeiro erro
       ├─ 01_etl.ipynb               validação → limpeza → atributos derivados → partição
       │                             → abre o ciclo (cycle_id, run pai no MLflow) → drift (PSI)
       ├─ 02_votacao_features.ipynb  SelectKBest + importância RF + RFE → votação (≥ 2 de 3)
       ├─ 03_treino_competicao.ipynb modelo A (LogisticRegression) vs modelo B (RandomForest)
       ├─ 04_selecao_promocao.ipynb  vencedor vs modelo vigente → promoção → versão + manifesto
       └─ 05_previsoes.ipynb         previsões com o modelo vigente + prova de reprodutibilidade
  │
  ├─ upload-artifact: evidências do run (notebooks executados, JSONs, previsões)
  └─ git commit + push: lotes, mlruns/, models/, reports/   (só se tudo acima passou)
```

`cycle_id` e `data_version` nascem no notebook 01 e ficam em `reports/ciclo_atual.json`. Cada notebook seguinte carrega esse arquivo com `load_cycle` e confere que os artefatos intermediários (dataset processado, atributos eleitos, competição, drift) pertencem ao mesmo ciclo. Um artefato velho interrompe o pipeline em vez de ser usado em silêncio.

Os notebooks orquestram e documentam. A lógica reutilizável fica em `src/`:

| Módulo | Responsabilidade |
|---|---|
| `src/utils.py` | Raiz do projeto, configuração, hashing, contexto do ciclo, MLflow (store portátil, run pai e runs filhos) |
| `src/ingestao.py` | Download verificado, partição em lotes, publicação de lote novo, `data_version` |
| `src/preprocessamento.py` | Validação por esquema, limpeza, outliers, atributos derivados, partição, `ColumnTransformer` |
| `src/features.py` | Os três seletores e a votação |
| `src/modelagem.py` | Registro de algoritmos, treino, métricas, escolha do vencedor, log no MLflow |
| `src/promocao.py` | Regra de promoção, versões imutáveis de modelo, manifesto |
| `src/drift.py` | PSI por atributo, status de drift, efeito na promoção |
| `src/pipeline.py` | Executor dos notebooks, resumo por ciclo e comparação entre ciclos |

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
# Ciclo completo com os lotes já publicados (notebooks 01 a 05)
python -m src.pipeline

# Simula a chegada de dados novos: publica o próximo lote e roda o ciclo
python -m src.pipeline --novo-lote

# Regenera reports/evidencias/comparacao_ciclos.md
python -m src.pipeline --comparar
```

Na primeira execução, o notebook 01 baixa a fonte da UCI e confere o SHA-256. Cada ciclo grava:

| Saída | Onde |
|---|---|
| Notebooks executados | `notebooks/` (sobrescritos) e `reports/evidencias/<cycle_id>/notebooks/` (cópia do ciclo) |
| Evidências por etapa | `reports/evidencias/<cycle_id>/01_etl.json`, `01_drift.json`, `02_votacao_features.csv/.json`, `03_competicao.json`, `04_promocao.json`, `05_previsoes.json` e `resumo_ciclo.json` |
| Comparação entre ciclos | `reports/evidencias/comparacao_ciclos.md` |
| Modelo vigente | `models/modelo_vigente.json`, `models/versoes/<model_id>/` e `models/historico_promocoes.jsonl` |
| Previsões | `reports/previsoes/previsoes_<cycle_id>.csv` |
| Rastreamento | `mlruns/mlflow.db` e `mlruns/artifacts/` |

Se qualquer notebook falhar, por erro de célula, timeout ou queda do kernel, o executor para e retorna código 1. Ele também:

- grava o notebook com a saída do erro;
- marca o run do ciclo no MLflow com `pipeline_status = falhou_em_<notebook>`;
- **restaura `models/modelo_vigente.json`** ao estado de antes do ciclo. Assim, uma promoção feita no notebook 04 não vale se o notebook 05 falhar. A versão gravada em `models/versoes/` permanece, mas deixa de ser a vigente.

Esse caminho foi testado forçando uma promoção e uma falha no notebook 05.

## MLflow

```bash
python -m src.pipeline --mlflow-ui      # http://127.0.0.1:5000
```

O comando abre a interface sobre `sqlite:///mlruns/mlflow.db`. Antes, ele ajusta os caminhos dos artefatos para o diretório atual (detalhes em [persistência](#github-actions-e-persistência)). Equivale a `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` com os caminhos já corrigidos.

Estrutura no experimento `credit-default-prediction`:

```text
run pai  <cycle_id>                 tags: cycle_id, data_version, lotes, pipeline_status, promotion_decision, active_model_id
  ├─ etl                            tamanhos, taxa de inadimplência por partição, relatório de validação
  ├─ drift                          PSI por atributo, status, limiares
  ├─ votacao_features               métodos, votos mínimos, tabela completa de votos, atributos eleitos
  ├─ modelo_A_logistic_regression   hiperparâmetros, random_state, atributos, val_* (F1, precisão, recall,
  ├─ modelo_B_random_forest         acurácia, ROC-AUC, tn/fp/fn/tp), tempo de treino, artefato do modelo
  ├─ promocao                       métrica do desafiante e do vigente, limiar aplicado, decisão, test_* do vigente
  └─ previsoes                      modelo usado, nº de previsões, CSV de previsões
```

Todo run filho tem as tags `cycle_id`, `data_version` e `stage`. Para comparar o modelo A com o B ou um ciclo com outro, filtre por `tags.stage = "treino"` e ordene por `metrics.val_f1`.

## Testes

```bash
pytest
```

São 37 testes com dados sintéticos, sem rede, que rodam em cerca de 2 segundos. Cobrem:
- a validação de dados: estrutura, alvo, nulos, códigos não documentados e limite de linhas inválidas;
- os atributos derivados e a estabilidade da partição;
- o ajuste do pré-processador só no treino;
- a votação de atributos: regra configurável e registro de cada voto;
- a competição: direção da métrica, desempate e mesmos atributos para A e B;
- a regra de promoção: inicial, margem, limite exato, desligada, drift e minimização;
- a geração do manifesto: campos, não sobrescrita e detecção de artefato adulterado;
- o PSI: contínuo, discreto e referência pelo manifesto;
- a `data_version`.

## Notebooks

A ordem de execução é obrigatória: cada um consome as saídas do anterior no mesmo ciclo.

| Notebook | Entrada | Saída |
|---|---|---|
| `01_etl.ipynb` | Lotes publicados | Dataset processado, `cycle_id`, `data_version`, drift |
| `02_votacao_features.ipynb` | Partição treino | Atributos eleitos e tabela de votos |
| `03_treino_competicao.ipynb` | Atributos eleitos, treino e validação | Dois modelos no MLflow, vencedor |
| `04_selecao_promocao.ipynb` | Vencedor, modelo vigente, drift | Decisão, nova versão (se promovido), avaliação final no teste |
| `05_previsoes.ipynb` | Modelo vigente, `novos_clientes.csv` | Previsões e prova de reprodutibilidade |

Para explorar de forma interativa, rode `jupyter lab` e execute os notebooks na ordem. Cada um explica objetivo, metodologia, entradas, saídas, decisões, resultados e conclusões.

## Configuração

Todos os valores ajustáveis ficam em `config/config.yaml`, e o código não repete nenhum deles:

| Seção | Conteúdo |
|---|---|
| `project` | Nome do projeto e do experimento MLflow |
| `data` | Fonte (URL, SHA-256), alvo, coluna de ID, `random_state: 42`, frações dos lotes, limite de linhas inválidas |
| `evaluation` | `primary_metric: f1`, `direction: maximize`, `test_size: 0.2`, `validation_size: 0.2`, métricas secundárias |
| `features` | `voting.minimum_votes: 2`, `k_best`, `rfe_features`, `random_forest_features` (10 cada), parâmetros dos estimadores internos dos seletores |
| `models` | Modelo A e modelo B: `enabled`, `label` e hiperparâmetros. A ordem define o desempate |
| `promotion` | `enabled: true`, `minimum_improvement: 0.01` |
| `paths` | Todos os caminhos, relativos à raiz |
| `drift` | `bins: 10`, `thresholds.moderate: 0.1`, `thresholds.severe: 0.25`, `relax_promotion_on_severe: true` |

Existe um único arquivo de configuração, em `config/config.yaml`. O esqueleto do enunciado também lista um `config.yaml` na raiz. Ele foi omitido de propósito, porque duas cópias divergiriam com o tempo.

## Protocolo de avaliação

Definido no notebook 01, antes de qualquer treino.

- **Partição por hash do ID.** `sha256("42:<ID>")` mapeia cada cliente para [0, 1]: abaixo de 0,2 vai para **teste**, entre 0,2 e 0,4 para **validação**, acima disso para **treino** (60/20/20).
- **Motivo.** Um `train_test_split` refeito a cada ciclo moveria clientes do teste antigo para o treino novo. Com o hash, cada cliente fica sempre na mesma partição. O teste continua isolado entre ciclos e o modelo vigente nunca treinou com a validação de um ciclo posterior.
- **Uso.**
  - Treino: ajusta o pré-processamento, os seletores e os modelos.
  - Validação: decide a competição e a promoção.
  - Teste: só é usado depois da decisão, para a avaliação final do modelo que ficou em produção (`active_test_*` no MLflow).
- **Sem vazamento.** O ETL aplica apenas transformações linha a linha. Imputação, one-hot e padronização ficam em um `ColumnTransformer` dentro do `Pipeline` de cada modelo, ajustado só no treino.
- **Métrica primária: F1, a ser maximizado.** Com 22% de positivos, um modelo que sempre prevê "adimplente" teria 78% de acurácia e F1 = 0. O F1 equilibra precisão (não negar crédito a bons pagadores) e recall (não deixar passar inadimplentes). Métricas secundárias: precisão, recall, acurácia, ROC-AUC e matriz de confusão.

## Votação de atributos

Os três métodos rodam no mesmo ciclo, sobre o treino imputado e padronizado:

| Método | Critério | Seleciona |
|---|---|---|
| 1. `SelectKBest(f_classif)` | Estatística F da ANOVA entre o atributo e a classe | 10 maiores |
| 2. Importância de `RandomForestClassifier` | Redução média de impureza de Gini | 10 maiores |
| 3. `RFE(LogisticRegression)` | Eliminação recursiva pelo menor coeficiente absoluto | 10 restantes |

**Regra:** um atributo é eleito com pelo menos `features.voting.minimum_votes` votos (padrão 2 de 3). Cada ciclo grava `reports/evidencias/<cycle_id>/02_votacao_features.csv`, com uma linha por (método, atributo) e as colunas `cycle_id`, `data_version`, `method`, `feature`, `score`, `rank`, `vote`, `votes` e `final_selection`. A mesma tabela vai para o MLflow. A votação usa as colunas originais, e não as colunas após o one-hot, para que o conjunto eleito seja legível e o mesmo nos dois modelos.

## Competição entre modelos

| | Modelo A | Modelo B |
|---|---|---|
| Algoritmo | `LogisticRegression` | `RandomForestClassifier` |
| Hiperparâmetros | `C=1.0`, `class_weight=balanced`, `max_iter=2000` | `n_estimators=150`, `max_depth=8`, `min_samples_leaf=20`, `class_weight=balanced` |
| Atributos | Os eleitos | Os mesmos eleitos |
| Treino / avaliação | Partição treino / partição validação | Idem |
| Semente | 42 | 42 |

Cada modelo é um `Pipeline(ColumnTransformer, estimador)`. O pré-processador é ajustado no treino e serializado junto com o modelo. O notebook 03 aborta se os dois modelos não tiverem exatamente os mesmos atributos e o mesmo protocolo. O vencedor é o de maior F1 na validação. Em empate, vence o primeiro em `config.models`.

## Promoção de modelo

```text
promover  ⇔  F1_desafiante(validação do ciclo) ≥ F1_vigente(validação do ciclo) + promotion.minimum_improvement
```

- O modelo vigente é **reavaliado na validação do ciclo atual**. Assim os dois são medidos na mesma régua. O F1 antigo do vigente veio de outra validação, menor.
- **Sem modelo vigente:** o vencedor é promovido como modelo inicial (`promovido_inicial`).
- **Drift severo** nos atributos do vigente: a margem cai para 0 (ver [drift](#monitoramento-de-drift-critério-10)).
- **`promotion.enabled: false`:** o vigente é sempre mantido.
- **Promovido:** o pacote (pipeline + atributos + metadados) vai para `models/versoes/<model_id>/modelo.joblib` com um `manifesto.json`, e `models/modelo_vigente.json` passa a apontar para ele. Um diretório de versão nunca é sobrescrito: gravar um `model_id` existente gera erro. O SHA-256 do artefato fica no manifesto e é conferido em toda carga.
- **Mantido:** o manifesto não muda.
- **Em todo ciclo:** a decisão e o motivo entram em `models/historico_promocoes.jsonl`, em `04_promocao.json` e no MLflow.
- **Recuperar uma versão anterior:** todas as versões promovidas continuam em `models/versoes/`. Todo modelo treinado, promovido ou não, continua no MLflow (`model_uri` no manifesto e nos runs de treino).

Exemplo de `models/modelo_vigente.json`: `model_id`, `run_id`, `model_uri`, `cycle_id`, `data_version`, `batches`, `algorithm`, `model_name`, `primary_metric`, `direction`, `metric_value`, `metric_partition`, `test_metrics`, `selected_features`, `hyperparameters`, `random_state`, `promotion_decision`, `decision_reason`, `artifact_path`, `artifact_sha256` e `timestamp`.

## Versionamento de dados e novos lotes

- **Fonte.** O arquivo original da UCI fica em `data/raw/source/`. É baixado uma vez, conferido por SHA-256 e **nunca alterado**. Não vai para o Git: o GitHub Actions o mantém em cache.
- **Lotes.** A fonte é embaralhada com `random_state` e dividida em `lote_001` (18.000 linhas), `lote_002` (6.000), `lote_003` (3.000) e `data/raw/inferencia/novos_clientes.csv` (3.000 linhas, sem alvo). Esse último arquivo é a base de previsões "nunca vistas".
- **Lotes publicados.** `data/raw/batches/` contém só os lotes já publicados. Eles vão para o Git e são o estado dos dados.
- **Novo lote.** `python -m src.pipeline --novo-lote` (ou `python -m src.ingestao --novo-lote`) publica o próximo lote. Quando todos já foram publicados, o ciclo roda com os existentes e registra um aviso.
- **`data_version`.** `"dv_" + sha256(nome:sha256(conteúdo) de cada lote publicado)[:16]`. Os mesmos lotes sempre geram a mesma versão, e qualquer lote novo ou alterado gera outra.
- **`cycle_id`.** Localmente, `ciclo_<UTC>_<8 caracteres da data_version>`. No GitHub Actions, `ciclo_gha_<run_id>_<tentativa>`.

Um lote novo muda o treino. Por isso, todo ciclo refaz a votação de atributos, o treino, a competição e a decisão de promoção.

## GitHub Actions e persistência

Existe um único workflow, `.github/workflows/pipeline_mlops.yml`:

- **Gatilho:** `workflow_dispatch` (execução manual em Actions → pipeline-mlops → Run workflow), com o input `novo_lote` que publica o próximo lote antes do ciclo.
- **Passos:** checkout, Python 3.12, `pip install`, cache da fonte UCI, `pytest`, `python -m src.pipeline [--novo-lote]`, resumo no *job summary*, `upload-artifact` das evidências e persistência no repositório.
- **Parada em falha:** cada passo para o job se falhar. O executor retorna código 1 no primeiro notebook com erro.
- **Concorrência:** um grupo `concurrency` impede dois ciclos simultâneos.

**Persistência.** O runner é descartável. Nada de um run anterior existe no disco do próximo. A estratégia é **o próprio repositório como armazenamento de estado**. Ao final de um ciclo bem-sucedido, o workflow faz commit e push de:

| Caminho | Conteúdo |
|---|---|
| `data/raw/batches/` | Lotes publicados (definem a `data_version`) |
| `mlruns/mlflow.db` + `mlruns/artifacts/` | Histórico completo do MLflow e artefatos de modelo |
| `models/` | Manifesto, versões promovidas e histórico de decisões |
| `reports/` | Evidências por ciclo, comparação e previsões |
| `notebooks/` | Notebooks com as saídas do último ciclo |

O próximo run começa com `actions/checkout` e encontra esse estado. Duas escolhas tornam isso seguro:

1. **Promoção atômica.** O passo de persistência só roda se todos os anteriores passaram. Se o notebook 05 falhar depois de uma promoção, o novo manifesto existe só no disco do runner descartado, e o modelo vigente no repositório não muda.
2. **Caminhos portáveis do MLflow.** O MLflow grava caminhos absolutos de artefato, e o runner clona o repositório em outro diretório. `configure_mlflow` reescreve o prefixo gravado no banco para o diretório atual antes de cada uso. Isso foi verificado movendo um store e recarregando um modelo logado.

Além disso, cada run publica as evidências como *artifact* do GitHub Actions, inclusive quando falha.

**Localmente** é o mesmo modelo, sem o commit automático. O estado fica em `mlruns/`, `models/` e `reports/`, e você versiona com `git` quando quiser.

Para ativar no GitHub: faça push do repositório e, em *Settings → Actions → General → Workflow permissions*, escolha **Read and write permissions**. O workflow usa o `GITHUB_TOKEN` padrão. Não há segredos no repositório.

## Monitoramento de drift (Critério 10)

**1. Que problema resolve.** Um modelo aprende a relação entre perfil e inadimplência em uma época. Se a população de entrada muda, por exemplo com uma nova política de limite, outro público ou uma crise econômica, os dados deixam de parecer com o treino. O desempenho cai sem que nada no código mude. Esse é o *data drift*, e ele só é percebido se for medido.

**2. Por que importa em crédito.** Concessão de crédito é um domínio clássico de drift. Ciclos econômicos mudam a capacidade de pagamento, e campanhas comerciais mudam o perfil de quem pede crédito. O PSI é o indicador padrão do setor para monitorar a estabilidade de *scorecards*.

**3. Como se integra ao pipeline.**

```text
lote novo → 01_etl: PSI por atributo (lotes do vigente vs lotes novos)
          → data/processed/drift.json + 01_drift.json + run "drift" no MLflow
          → 04_selecao_promocao: drift severo nos atributos do vigente ⇒ margem de promoção = 0
```

- **Referência:** os lotes com que o modelo vigente foi treinado, registrados no manifesto.
- **Atual:** os lotes publicados depois.
- **Faixas:** atributos contínuos usam 10 faixas por quantis da referência; atributos discretos usam um valor por faixa.
- **Status do ciclo:** o pior status entre os atributos que o modelo vigente consome.

**Decisão.** Com drift severo, o vigente foi treinado em uma população que não existe mais. Exigir que o desafiante supere o vigente por 1 ponto de F1 protegeria um modelo obsoleto. Basta que o desafiante não seja pior.

**4. Limiares** (`drift.thresholds`):

| PSI | Status |
|---|---|
| < 0,10 | estável |
| 0,10 a 0,25 | moderado |
| ≥ 0,25 | severo |

**5. Evidência de que funciona.**
- `01_drift.json` e o run `drift` no MLflow de cada ciclo.
- Os lotes deste projeto são partições aleatórias da mesma fonte, então o resultado esperado e observado é *estável*. Isso mostra que o detector não dispara alarme falso.
- O notebook 01 traz um **controle positivo**. Ele aplica um deslocamento conhecido a uma cópia da validação: limite 40% menor, um mês a mais de atraso, clientes 8 anos mais velhos. O PSI classifica esses atributos como *severo*, e a mesma amostra sem deslocamento como *estável*. As duas classificações são asserções que param o pipeline se falharem.
- Testes automatizados cobrem atributos contínuos e discretos, a escolha da referência pelo manifesto e o efeito na regra de promoção.

## Demonstração em dois ciclos

Os dois ciclos foram executados localmente, com `python -m src.pipeline` e depois `python -m src.pipeline --novo-lote`. As evidências completas estão em `reports/evidencias/`, e a tabela abaixo resume `reports/evidencias/comparacao_ciclos.md`.

| | Ciclo 1 | Ciclo 2 |
|---|---|---|
| `cycle_id` | `ciclo_20261116T223751Z_a2d3d2d2` | `ciclo_20261116T223819Z_2db70707` |
| `data_version` | `dv_a2d3d2d25ce5c5c8` | `dv_2db70707140b1359` |
| Lotes | `lote_001` | `lote_001` + `lote_002` |
| Linhas (treino / validação / teste) | 18.000 (10.851 / 3.613 / 3.536) | 24.000 (14.475 / 4.819 / 4.706) |
| Drift | sem referência (não havia modelo vigente) | estável (maior PSI: 0,0033 em `AGE`) |
| Atributos eleitos | 9, incluindo `PAY_4` | 9, `PAY_4` sai e `PAY_6` entra |
| Modelo A (LogReg) F1 validação | 0,5293 | 0,5295 |
| Modelo B (RF) F1 validação | 0,5439 | 0,5410 |
| Vencedor | Modelo B | Modelo B |
| Vigente reavaliado na validação | - | 0,5403 |
| Decisão | `promovido_inicial` | `mantido`: 0,5410 < 0,5403 + 0,01 |
| Modelo vigente | RF do ciclo 1 | RF do ciclo 1 (preservado) |
| F1 no teste do modelo vigente | 0,5249 | 0,5363 |
| Previsões (3.000 novos clientes) | reproduzíveis | reproduzíveis |

**O que mudou entre os ciclos.**
- A chegada do `lote_002` gerou uma nova `data_version` e um novo `cycle_id`.
- A votação rodou de novo e trocou um atributo.
- Os dois modelos foram retreinados e comparados de novo.
- O vencedor do ciclo 2 superou o vigente por apenas 0,0007 de F1, abaixo da margem de 0,01. Por isso a promoção foi **recusada** e o modelo do ciclo 1 continua em produção.
- O F1 de teste do mesmo modelo mudou de 0,5249 para 0,5363 porque o conjunto de teste cresceu com os clientes do `lote_002`. Os clientes de teste do ciclo 1 continuam todos no teste.
- O controle positivo de drift do notebook 01 classificou `LIMIT_BAL`, `PAY_0`, `AGE` e `UTILIZACAO_ATUAL` deslocados como severos (PSI de 0,54 a 2,25). Os mesmos atributos sem deslocamento ficaram estáveis (PSI ≤ 0,0035).
- Os dois ciclos rodaram duas vezes nesta máquina e produziram as mesmas métricas, os mesmos votos e as mesmas decisões até a 4ª casa decimal.

O ciclo 3 (`lote_003`) fica para a primeira execução do workflow no GitHub Actions.

## Reprodutibilidade

```bash
git clone <url-do-repositorio>
cd mlops-credit-default-prediction
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m src.pipeline              # novo ciclo sobre o estado versionado
python -m src.pipeline --mlflow-ui  # inspecionar o histórico
```

Para reproduzir do zero, sem o histórico versionado, apague o estado e rode os dois ciclos:

```bash
rm -rf mlruns/mlflow.db mlruns/artifacts models/versoes models/modelo_vigente.json models/historico_promocoes.jsonl \
       reports/evidencias/ciclo_* reports/previsoes data/raw/batches/lote_00[2-9].csv
python -m src.pipeline              # ciclo 1: lote_001
python -m src.pipeline --novo-lote  # ciclo 2: lote_001 + lote_002
```

O que garante o mesmo resultado em outra máquina:
- o checksum da fonte;
- a semente fixa (42) na partição em lotes, na partição treino/validação/teste, nos seletores e nos modelos;
- a `data_version` derivada do conteúdo;
- as dependências com versão fixa;
- os caminhos relativos à raiz do projeto.

Os `cycle_id` locais incluem o horário de execução, então diferem entre execuções. Métricas, votos e decisões se repetem.

## Limitações conhecidas

- **Lotes simulados.** Como a fonte é estática, a demonstração não tem drift real. A evidência de detecção vem do controle positivo e dos testes.
- **Crescimento do repositório.** Cada ciclo adiciona cerca de 6 MB, quase tudo o artefato da Random Forest no MLflow (formato skops). Para um trabalho acadêmico com execução semanal isso é aceitável. Em produção, `mlruns/` iria para um armazenamento de objetos ou um servidor MLflow.
- **Limiar de decisão.** O limiar de classificação é fixo em 0,5 (com `class_weight=balanced`). Otimizá-lo na validação poderia elevar o F1, mas exigiria uma segunda validação para não enviesar a competição.
- **Viés leve a favor do desafiante.** O F1 do desafiante é o melhor de dois modelos escolhidos na mesma validação usada para compará-lo ao vigente, então é levemente otimista. A margem de 0,01 também serve para absorver esse viés. Uma alternativa mais rigorosa seria validação cruzada do vencedor no treino.
- **Workflow não executado aqui.** O workflow foi validado com `actionlint`, mas só roda de fato depois do push para o GitHub.
