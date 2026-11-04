# Projeto N3 · Pipeline MLOps para previsão de cancelamento de reservas

Trabalho avaliativo N3 de **Análise Preditiva** (Católica de Santa Catarina, Eng. de Software e Eng. de IA, Prof. Rodrigo Hagstrom).

Pipeline preditivo versionado e agendado que, a cada ciclo, incorpora dados novos, elege features por votação de três métodos, faz dois algoritmos competirem, registra tudo no MLflow e só promove um novo modelo se ele superar o vigente por uma regra definida antes.

**Problema:** prever se uma reserva de hotel será cancelada (classificação binária), usando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Detalhes e decisões em [`docs/etapa1_problema_dados_avaliacao.md`](docs/etapa1_problema_dados_avaliacao.md).

## Status

| Etapa | Critério | Responsável | Status |
|---|---|---|---|
| 1. Problema, dados e avaliação | 1 | Gabriel | ✅ pronto |
| 2. ETL e engenharia de atributos | 2 | Gabriel | ✅ pronto (testes + 2 ciclos de evidência) |
| 3. Três métodos de seleção | 3 | a definir | ⬜ esqueleto (`src/features.py`, notebook 02) |
| 4. Votação e eleição das features | 4 | a definir | ⬜ esqueleto |
| 5. Dois algoritmos disputam | 5 | a definir | ⬜ esqueleto (`src/modelagem.py`, notebook 03) |
| 6. MLflow e rastreabilidade | 6 | a definir | ⬜ config proposta, sem código |
| 7. Workflow único e agendamento | 7 | a definir | 🟨 esqueleto roda testes e o notebook 01; **sem `schedule` e nunca executado no Actions** |
| 8. Eleição, promoção e previsões | 8 | a definir | ⬜ esqueleto (`src/promocao.py`, notebooks 04 e 05) |
| 9. Repositório, relatório e evidências | 9 | Gabriel (coord.) | 🟨 README e histórico prontos; falta o relatório PDF |
| 10. Recurso adicional | 10 | a definir | ⬜ ideia sugerida em `docs/divisao_de_tarefas.md` |

Quem faz o quê e como contribuir: [`docs/divisao_de_tarefas.md`](docs/divisao_de_tarefas.md). Histórico de commits por integrante: [`HISTORICO_DESENVOLVIMENTO.md`](HISTORICO_DESENVOLVIMENTO.md).

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução local

```bash
python -m pytest -q                      # testes (ingestão, ETL, ausência de vazamento, avaliação)
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
data/raw/                              lotes incorporados (*.csv.gz) + manifest.json
data/incoming/                         lotes futuros, liberados um a um para simular dados novos
data/processed/                        saídas intermediárias do ciclo (não versionadas)
docs/                                  decisões da Etapa 1 e divisão de tarefas
notebooks/01_etl.ipynb ... 05_previsoes.ipynb
src/                                   ingestao, preprocessamento, avaliacao (prontos); features, modelagem, promocao (esqueletos)
tests/                                 pytest
models/modelo_vigente.json             gerado pela promoção (ainda não existe)
evidencias/                            diagnósticos e notebooks executados
scripts/                               preparar_lotes.py, gerar_historico.py
```

## Contrato entre notebooks

O notebook 01 grava `data/processed/ciclo_atual.json` (com `cycle_id`, `data_version`, caminhos e colunas), `dev.parquet` e `teste.parquet`. Os demais apenas leem, via `src.config.carregar_contexto`. O conjunto de teste só é aberto na avaliação final. Cada ciclo carrega o mesmo `cycle_id` e `data_version` do começo ao fim.

## Execução agendada e MLflow

Ainda não implementados (critérios 7 e 6). O esqueleto do workflow lista as pendências. Lembrete: o runner do GitHub Actions é efêmero, então o histórico do MLflow, o manifesto dos lotes e o modelo vigente precisam de armazenamento persistente entre ciclos.

## Dados e licença

Antonio, Almeida & Nunes (2019). *Hotel booking demand datasets*. Data in Brief 22:41-49. DOI 10.1016/j.dib.2018.11.126. Licença **CC BY** (artigo de dados original). Nenhum segredo ou dado restrito é versionado.
