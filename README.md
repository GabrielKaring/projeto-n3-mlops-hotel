# Projeto N3 · Pipeline MLOps para previsão de cancelamento de reservas

Trabalho avaliativo N3 de **Análise Preditiva** (Católica de Santa Catarina, Eng. de Software e Eng. de IA, Prof. Rodrigo Hagstrom).

Pipeline preditivo versionado e agendado que, a cada ciclo, incorpora dados novos, elege features por votação de três métodos, faz dois algoritmos competirem, registra tudo no MLflow e só promove um novo modelo se ele superar o vigente por uma regra definida antes.

**Problema:** prever se uma reserva de hotel será cancelada (classificação binária), usando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Detalhes e decisões em [`docs/etapa1_problema_dados_avaliacao.md`](docs/etapa1_problema_dados_avaliacao.md).

## Status por feature

Cada feature é um critério do enunciado e tem um dono, ou dois quando dividida (1 a 4 features por pessoa), definido em [`config/equipe.yaml`](config/equipe.yaml).

| Feature | Critério | Dono | Status |
|---|---|---|---|
| F1 | 1 · Problema, dados e avaliação | Gabriel Karing | ✅ pronto |
| F2 | 2 · ETL e engenharia de atributos | Gabriel Karing | ✅ pronto (testes + 2 ciclos de evidência) |
| F3 | 3 · Três métodos de seleção | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/selecao/`) |
| F4 | 4 · Votação e eleição das features | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/votacao/`) |
| F5 | 5 · Dois algoritmos disputam | Gabriel Wendorff | ✅ pronto (testes + evidência em `evidencias/competicao/`) |
| F6 | 6 · MLflow e rastreabilidade | José | ✅ pronto (histórico persistido entre execuções na branch `estado-pipeline`) |
| F7 | 7 · Workflow único e agendamento | Diego | ✅ pronto (`schedule` + `workflow_dispatch`, ver [`docs/f7_workflow.md`](docs/f7_workflow.md)) |
| F8 | 8 · Eleição, promoção e previsões | Diego | ✅ pronto (testes + decisões em `models/modelo_vigente.json`) |
| F9 | 9 · Repositório, relatório e evidências | Vinicius | ✅ pronto ([`reports/relatorio_n3.pdf`](reports/relatorio_n3.pdf)) |
| F10 | 10 · Recurso adicional (drift com alerta) | Lucas e Guilherme | ✅ pronto (passo do workflow logo depois do ETL, ver [`docs/f10_drift.md`](docs/f10_drift.md)) |

Definição de pronto, evidência e teste de cada feature: [`docs/divisao_de_tarefas.md`](docs/divisao_de_tarefas.md). Commits seguem `[Fn] descrição`; o [`HISTORICO_DESENVOLVIMENTO.md`](HISTORICO_DESENVOLVIMENTO.md) mostra, por feature, quem fez o quê.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Testado com Python 3.13, pandas 2.3, scikit-learn 1.9 e mlflow 3.18 (o que o `requirements.txt` instala em ambiente novo). O pandas está limitado a `<3` porque o mlflow 3.x exige isso; sem o teto o `pip install` trava. O workflow usa Python 3.12.

## Execução local

```bash
python -m pytest -q                      # testes (ingestão, ETL, seleção de features, votação, ausência de vazamento, avaliação, MLflow, divisão da equipe, histórico)
python -m src.ingestao                   # registra lotes novos em data/raw e mostra data_version
python -m src.ingestao --liberar 1       # simula a chegada do próximo lote mensal (data/incoming -> data/raw)
python -m src.drift --historico          # drift do lote mais recente contra a referência (F10), saídas em evidencias/drift/
```

Um ciclo completo, na mesma ordem do workflow (os notebooks executados vão para `evidencias/execucao/`):

```bash
python -m src.ingestao --liberar 1
bash scripts/executar_notebook.sh 01_etl
python -m src.drift --historico
bash scripts/executar_notebook.sh 02_votacao_features
bash scripts/executar_notebook.sh 03_treino_competicao
bash scripts/executar_notebook.sh 04_selecao_promocao
bash scripts/executar_notebook.sh 05_previsoes
python -m src.rastreio
```

No Windows, rode o `bash` pelo Git Bash ou WSL. Leva uns 2 minutos por ciclo.

`--liberar` move um arquivo de lugar. Para voltar ao estado inicial: `git restore data && git clean -fd data/raw`.

Os dados já estão no repositório, divididos em lotes mensais. Para regerá-los a partir do CSV original do Kaggle: `python scripts/preparar_lotes.py hotel_bookings.csv`.

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     métrica, partição, ETL, votação, modelos, promoção, MLflow
config/equipe.yaml                     dono de cada feature (F1..F10), validado por tests/test_equipe.py
data/raw/                              lotes incorporados (*.csv.gz) + manifest.json
data/incoming/                         lotes futuros, liberados um a um para simular dados novos
data/processed/                        saídas intermediárias do ciclo (não versionadas)
docs/                                  decisões de cada feature, divisão de tarefas e gates
notebooks/01_etl.ipynb ... 05_previsoes.ipynb
src/                                   ingestao, preprocessamento, avaliacao, features, votacao, modelagem, rastreio, promocao, drift
tests/                                 pytest
models/modelo_vigente.json             vigente e histórico de decisões (gerado pelo notebook 04)
models/registro/                       vencedor de cada ciclo (no Actions, fica na Release `modelos`)
mlflow_estado/                         histórico do MLflow: mlflow.db + artefatos/ (não versionado; ver docs/f6_mlflow.md)
evidencias/                            diagnósticos, notebooks executados, drift, decisões e previsões
reports/relatorio_n3.pdf               relatório final
scripts/                               preparar_lotes.py, gerar_historico.py, executar_notebook.sh
```

## Contrato entre notebooks

O notebook 01 grava `data/processed/ciclo_atual.json` (com `cycle_id`, `data_version`, caminhos e colunas), `dev.parquet` e `teste.parquet`. Os demais apenas leem, via `src.config.carregar_contexto`. O conjunto de teste só é aberto na avaliação final. Cada ciclo carrega o mesmo `cycle_id` e `data_version` do começo ao fim.

## MLflow

Cada notebook registra seus runs no experimento `n3-hotel-cancelamento`, sempre com as tags `cycle_id` e `data_version`. O histórico fica em `mlflow_estado/` (SQLite + artefatos). Detalhes em [`docs/f6_mlflow.md`](docs/f6_mlflow.md).

```bash
python -m src.rastreio                                           # lista ciclos e runs; exporta evidencias/mlflow/historico_runs.csv
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db  # interface em http://127.0.0.1:5000
```

## Execução agendada

O workflow [`.github/workflows/pipeline_mlops.yml`](.github/workflows/pipeline_mlops.yml) roda um ciclo por execução: recupera o estado, libera 1 lote novo, roda o notebook 01, o drift e os notebooks 02 a 05, e guarda o estado de novo. Dispara pelo `schedule` (todo dia às 06:00 de Brasília) ou manualmente em *Actions → pipeline-mlops → Run workflow*.

Como o runner é apagado no fim, o estado fica fora dele:

- branch `estado-pipeline`: histórico do MLflow, manifesto dos lotes, modelo vigente e previsões;
- Release `modelos`: o vencedor de cada ciclo (nada é apagado ou substituído);
- artefato `evidencias-<run_id>` de cada execução: evidências do ciclo.

Para abrir esse estado na sua máquina e consultar o MLflow:

```bash
git fetch origin estado-pipeline
git archive origin/estado-pipeline | tar -xf -
python -m src.rastreio
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db
```

Isso sobrescreve o `data/raw/manifest.json` local. Para voltar: `git checkout -- data/raw/manifest.json`. Detalhes, falha simulada e limitações em [`docs/f7_workflow.md`](docs/f7_workflow.md).

## Previsões

O notebook 05 carrega o vigente (Pipeline completo: pré-processamento, features eleitas e algoritmo) e prevê o próximo lote de `data/incoming/`, que ainda não participou da escolha do modelo. As previsões ficam em `evidencias/previsoes/previsoes_<cycle_id>.csv`, com o ciclo e o modelo que previu. Quando esse lote chega no ciclo seguinte, o próprio notebook confere as previsões com o rótulo real.

Para prever com o modelo vigente localmente, rode o ciclo completo acima (ou abra o estado do Actions e baixe o vigente com `gh release download modelos --dir models/registro`) e depois `bash scripts/executar_notebook.sh 05_previsoes`.

## Dados e licença

Antonio, Almeida & Nunes (2019). *Hotel booking demand datasets*. Data in Brief 22:41-49. DOI 10.1016/j.dib.2018.11.126. Licença **CC BY** (artigo de dados original). Nenhum segredo ou dado restrito é versionado.
