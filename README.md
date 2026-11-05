# Projeto N3 · Pipeline MLOps para previsão de cancelamento de reservas

Trabalho avaliativo N3 de **Análise Preditiva** (Católica de Santa Catarina, Eng. de Software e Eng. de IA, Prof. Rodrigo Hagstrom).

Pipeline preditivo versionado e agendado que, a cada ciclo, incorpora dados novos, elege features por votação de três métodos, faz dois algoritmos competirem, registra tudo no MLflow e só promove um novo modelo se ele superar o vigente por uma regra definida antes.

**Problema:** prever se uma reserva de hotel será cancelada (classificação binária), usando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Detalhes e decisões em [`docs/etapa1_problema_dados_avaliacao.md`](docs/etapa1_problema_dados_avaliacao.md).

## Status por feature

Cada feature é um critério do enunciado e tem um dono (1 a 3 features por pessoa), definido em [`config/equipe.yaml`](config/equipe.yaml).

| Feature | Critério | Dono (proposta) | Status |
|---|---|---|---|
| F1 | 1 · Problema, dados e avaliação | Gabriel Karing | ✅ pronto |
| F2 | 2 · ETL e engenharia de atributos | Gabriel Karing | ✅ pronto (testes + 2 ciclos de evidência) |
| F3 | 3 · Três métodos de seleção | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/selecao/`) |
| F4 | 4 · Votação e eleição das features | Gabriel Wendorff | ⬜ esqueleto (`src/votacao.py`, notebook 02) |
| F5 | 5 · Dois algoritmos disputam | Guilherme | ⬜ esqueleto (`src/modelagem.py`, notebook 03) |
| F6 | 6 · MLflow e rastreabilidade | José | ⬜ esqueleto (`src/rastreio.py`) e config proposta |
| F7 | 7 · Workflow único e agendamento | José | 🟨 esqueleto roda testes e o notebook 01; **sem `schedule` e nunca executado no Actions** |
| F8 | 8 · Eleição, promoção e previsões | Lucas | ⬜ esqueleto (`src/promocao.py`, notebooks 04 e 05) |
| F9 | 9 · Repositório, relatório e evidências | Vinicius | 🟨 README e histórico prontos; falta o relatório PDF |
| F10 | 10 · Recurso adicional | Diego | ⬜ ideia sugerida em `docs/divisao_de_tarefas.md` |

Definição de pronto, evidência e teste de cada feature: [`docs/divisao_de_tarefas.md`](docs/divisao_de_tarefas.md). Commits seguem `[Fn] descrição`; o [`HISTORICO_DESENVOLVIMENTO.md`](HISTORICO_DESENVOLVIMENTO.md) mostra, por feature, quem fez o quê.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Testado com Python 3.13, pandas 2.3, scikit-learn 1.9 e mlflow 3.18 (o que o `requirements.txt` instala em ambiente novo). O pandas está limitado a `<3` porque o mlflow 3.x exige isso; sem o teto o `pip install` trava. O workflow usa Python 3.12, ainda não testado no runner do GitHub.

## Execução local

```bash
python -m pytest -q                      # testes (ingestão, ETL, seleção de features, ausência de vazamento, avaliação, divisão da equipe, histórico)
python -m src.ingestao                   # registra lotes novos em data/raw e mostra data_version
python -m src.ingestao --liberar 1       # simula a chegada do próximo lote mensal (data/incoming -> data/raw)
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
src/                                   ingestao, preprocessamento, avaliacao, features (prontos); votacao, modelagem, rastreio, promocao (esqueletos)
tests/                                 pytest
models/modelo_vigente.json             gerado pela promoção (ainda não existe)
evidencias/                            diagnósticos e notebooks executados
scripts/                               preparar_lotes.py, gerar_historico.py
```

## Contrato entre notebooks

O notebook 01 grava `data/processed/ciclo_atual.json` (com `cycle_id`, `data_version`, caminhos e colunas), `dev.parquet` e `teste.parquet`. Os demais apenas leem, via `src.config.carregar_contexto`. O conjunto de teste só é aberto na avaliação final. Cada ciclo carrega o mesmo `cycle_id` e `data_version` do começo ao fim.

## Execução agendada e MLflow

Ainda não implementados (features F7 e F6). O esqueleto do workflow lista as pendências. Lembrete: o runner do GitHub Actions é efêmero, então o histórico do MLflow, o manifesto dos lotes e o modelo vigente precisam de armazenamento persistente entre ciclos.

## Dados e licença

Antonio, Almeida & Nunes (2019). *Hotel booking demand datasets*. Data in Brief 22:41-49. DOI 10.1016/j.dib.2018.11.126. Licença **CC BY** (artigo de dados original). Nenhum segredo ou dado restrito é versionado.
