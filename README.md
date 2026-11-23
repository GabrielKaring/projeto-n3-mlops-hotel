# Projeto N3 · Pipeline MLOps para previsão de cancelamento de reservas

Trabalho avaliativo N3 de **Análise Preditiva** (Católica de Santa Catarina, Eng. de Software e Eng. de IA, Prof. Rodrigo Hagstrom).

Pipeline preditivo versionado e agendado que, a cada ciclo, incorpora dados novos, elege features por votação de três métodos, faz dois algoritmos competirem, registra tudo no MLflow e só promove um novo modelo se ele superar o vigente por uma regra definida antes.

**Problema:** prever se uma reserva de hotel será cancelada (classificação binária), usando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Detalhes e decisões em [`docs/etapa1_problema_dados_avaliacao.md`](docs/etapa1_problema_dados_avaliacao.md).

## Status por feature

Cada feature é um critério do enunciado e tem um dono, ou dois quando dividida (1 a 4 features por pessoa), definido em [`config/equipe.yaml`](config/equipe.yaml).

| Feature | Critério | Dono (proposta) | Status |
|---|---|---|---|
| F1 | 1 · Problema, dados e avaliação | Gabriel Karing | ✅ pronto |
| F2 | 2 · ETL e engenharia de atributos | Gabriel Karing | ✅ pronto (testes + 2 ciclos de evidência) |
| F3 | 3 · Três métodos de seleção | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/selecao/`) |
| F4 | 4 · Votação e eleição das features | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/votacao/`) |
| F5 | 5 · Dois algoritmos disputam | Gabriel Wendorff | ⬜ esqueleto (`src/modelagem.py`, notebook 03) |
| F6 | 6 · MLflow e rastreabilidade | José | 🟨 registro, consulta e testes prontos (`src/rastreio.py`, evidência em `evidencias/mlflow/`); falta a persistência no Actions, combinada com a F7 |
| F7 | 7 · Workflow único e agendamento | Diego | 🟨 esqueleto roda testes e o notebook 01; **sem `schedule` e nunca executado no Actions** |
| F8 | 8 · Eleição, promoção e previsões | Diego | ⬜ esqueleto (`src/promocao.py`, notebooks 04 e 05) |
| F9 | 9 · Repositório, relatório e evidências | Vinicius | 🟨 README e histórico prontos; falta o relatório PDF |
| F10 | 10 · Recurso adicional (drift com alerta) | Lucas e Guilherme | 🟨 regra, `src/drift.py` e testes na branch `f10-drift` ([`docs/f10_drift.md`](docs/f10_drift.md)); falta integrar no workflow e gerar evidências |

Definição de pronto, evidência e teste de cada feature: [`docs/divisao_de_tarefas.md`](docs/divisao_de_tarefas.md). Commits seguem `[Fn] descrição`; o [`HISTORICO_DESENVOLVIMENTO.md`](HISTORICO_DESENVOLVIMENTO.md) mostra, por feature, quem fez o quê.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Testado com Python 3.13, pandas 2.3, scikit-learn 1.9 e mlflow 3.18 (o que o `requirements.txt` instala em ambiente novo). O pandas está limitado a `<3` porque o mlflow 3.x exige isso; sem o teto o `pip install` trava. O workflow usa Python 3.12, ainda não testado no runner do GitHub.

## Execução local

```bash
python -m pytest -q                      # testes (ingestão, ETL, seleção de features, votação, ausência de vazamento, avaliação, MLflow, divisão da equipe, histórico)
python -m src.ingestao                   # registra lotes novos em data/raw e mostra data_version
python -m src.ingestao --liberar 1       # simula a chegada do próximo lote mensal (data/incoming -> data/raw)
python -m src.drift --historico          # drift do lote mais recente contra a referência (F10), saídas em evidencias/drift/
cd notebooks && jupyter nbconvert --to notebook --execute 01_etl.ipynb --output 01_executado.ipynb --output-dir ../evidencias/execucao
```

`--liberar` move um arquivo de lugar. Para voltar ao estado inicial: `git restore data && git clean -fd data/raw`.

Os dados já estão no repositório, divididos em lotes mensais. Para regerá-los a partir do CSV original do Kaggle: `python scripts/preparar_lotes.py hotel_bookings.csv`.

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (esqueleto)
config/config.yaml                     métrica, partição, ETL, votação, modelos, promoção, MLflow
config/equipe.yaml                     dono de cada feature (F1..F10), validado por tests/test_equipe.py
data/raw/                              lotes incorporados (*.csv.gz) + manifest.json
data/incoming/                         lotes futuros, liberados um a um para simular dados novos
data/processed/                        saídas intermediárias do ciclo (não versionadas)
docs/                                  decisões da Etapa 1 e divisão de tarefas
notebooks/01_etl.ipynb ... 05_previsoes.ipynb
src/                                   ingestao, preprocessamento, avaliacao, features, votacao, rastreio (prontos); modelagem, promocao (esqueletos)
tests/                                 pytest
models/modelo_vigente.json             gerado pela promoção (ainda não existe)
mlflow_estado/                         histórico do MLflow: mlflow.db + artefatos/ (não versionado; ver docs/f6_mlflow.md)
evidencias/                            diagnósticos e notebooks executados
scripts/                               preparar_lotes.py, gerar_historico.py
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

Ainda não implementada (feature F7). O esqueleto do workflow lista as pendências. Lembrete: o runner do GitHub Actions é efêmero, então o histórico do MLflow (`mlflow_estado/`), o manifesto dos lotes e o modelo vigente precisam de armazenamento persistente entre ciclos. A F6 propõe uma branch `estado-pipeline` em [`docs/f6_mlflow.md`](docs/f6_mlflow.md).

## Dados e licença

Antonio, Almeida & Nunes (2019). *Hotel booking demand datasets*. Data in Brief 22:41-49. DOI 10.1016/j.dib.2018.11.126. Licença **CC BY** (artigo de dados original). Nenhum segredo ou dado restrito é versionado.
