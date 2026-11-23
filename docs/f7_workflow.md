# F7 · Workflow único e agendamento

Dono: Diego. Workflow em [`.github/workflows/pipeline_mlops.yml`](../.github/workflows/pipeline_mlops.yml), execução de cada notebook em [`scripts/executar_notebook.sh`](../scripts/executar_notebook.sh), testes em [`tests/test_workflow.py`](../tests/test_workflow.py).

## Gatilhos

| Gatilho | Quando | Para quê |
|---|---|---|
| `schedule` | Todo dia às 09:00 UTC (06:00 em Brasília) | Execução oficial. O enunciado exige evidência deste gatilho |
| `workflow_dispatch` | Manual, na aba Actions | Depuração. A entrada `falhar_em` simula uma falha antes de um notebook |

- O `schedule` só dispara a partir da branch padrão. O workflow precisa estar na `main`.
- O GitHub pode atrasar ou pular disparos agendados em horários de pico. Deixem folga antes da entrega.
- Cada execução libera 1 lote de `data/incoming/` (são 8, de jan a ago/2017). Com o disparo diário, o último lote (ago/2017) entra na execução de 27/11/2026, e a de 26/11 é a última com um lote futuro para prever. Depois disso, os ciclos seguem rodando com os mesmos dados; o notebook 05 avisa que não há lote novo, mostra as últimas previsões feitas e roda a demonstração com reservas novas (`src/prever.py`). A decisão esperada nesses ciclos é manter o vigente. O cron não foi alterado, para não mudar a simulação de chegada de dados; para mostrar um ciclo com lote novo depois de 27/11, rode um ciclo local (README, "Execução local").
- As actions usam as versões com Node 24 (`actions/checkout@v7`, `actions/setup-python@v7`, `actions/upload-artifact@v7`), atualizadas em 23/11/2026 sem mudar a lógica dos passos.

## O que cada execução faz

| # | Passo | Falha interrompe? |
|---|---|---|
| 1 | Instala as dependências e roda `pytest` | sim |
| 2 | Recupera o estado da branch `estado-pipeline` | sim. Se não conseguir consultar a branch, para em vez de começar do zero |
| 3 | Devolve para `data/raw/` os lotes que ciclos anteriores já liberaram | sim |
| 4 | Baixa o modelo vigente da Release `modelos` | sim. Sem ele não há comparação justa |
| 5 | `python -m src.ingestao --liberar 1`: chega um lote novo | sim |
| 6 | Notebooks 01 → 02 → 03 → 04, um por passo | sim |
| 7 | Envia o vencedor do ciclo para a Release `modelos` | sim |
| 8 | Notebook 05 (previsões no próximo lote) | sim |
| 9 | Exporta o histórico do MLflow (`python -m src.rastreio`) | roda mesmo após falha |
| 10 | Guarda o estado na branch `estado-pipeline` | roda mesmo após falha (ver abaixo) |
| 11 | Publica as evidências como artefato da execução | roda mesmo após falha |

Os notebooks rodam com `jupyter nbconvert --execute`, sem a opção que tolera erros. Um erro em qualquer célula falha o passo, e os passos seguintes não rodam.

## Estado entre execuções

O runner do Actions é apagado no fim de cada execução. O estado fica em dois lugares:

| Onde | O quê | Quem produz |
|---|---|---|
| Branch `estado-pipeline` | `mlflow_estado/` (banco SQLite e artefatos do MLflow) | F6 |
| | `data/raw/manifest.json` (lotes já incorporados) | F2 |
| | `models/modelo_vigente.json` (vigente e histórico de decisões) | F8 |
| | `evidencias/previsoes/` (para conferir as previsões quando o lote chega) | F8 |
| | `evidencias/etl/diagnostico_*.json` (para o notebook 01 saber o que mudou) | F2 |
| Release `modelos` | `modelo_<cycle_id>_<A\|B>.joblib`: o vencedor de cada ciclo, comprimido | F8 |

**O que é gravado quando o ciclo falha:**
- O `mlflow_estado/` é gravado sempre, porque os runs de ciclos que falharam também são histórico.
- O manifesto dos lotes, o vigente e os diagnósticos só são gravados se o ciclo chegou até a Release. Se falhar antes, o próximo ciclo parte do estado anterior e libera o mesmo lote de novo. Assim, uma falha nunca deixa um vigente promovido sem o arquivo do modelo.
- As previsões só são gravadas se o notebook 05 terminou.
- Na Release, nenhum asset é apagado ou substituído: o envio falha se o nome já existe.

**Tamanho:** `mlflow_estado/` vai inteiro, incluindo os `.joblib` que o notebook 03 anexa ao MLflow. Os candidatos são salvos comprimidos (o B fica em uns 22 MB), então a branch `estado-pipeline` cresce perto de 22 MB por ciclo, abaixo do aviso de 50 MB por arquivo do GitHub.

**Execuções locais:** as evidências `*-local` versionadas na `main` são ignoradas no runner, para não se misturarem com os ciclos do Actions.

## Requisitos no repositório

- *Settings → Actions → General → Workflow permissions* = **Read and write permissions**. Sem isso, o workflow não grava na branch `estado-pipeline` nem cria a Release.
- O workflow declara `permissions: contents: write` e `concurrency` (nunca dois ciclos juntos, porque o SQLite aceita um escritor por vez).

## Como demonstrar uma falha (gate G7)

Na aba Actions → `pipeline-mlops` → *Run workflow*, escolha `falhar_em = 03_treino_competicao`, por exemplo. O passo do notebook 03 falha, o 04 e a Release não rodam, e o estado grava só o MLflow. O vigente continua o mesmo.

## Como abrir o estado do Actions localmente

```bash
git fetch origin estado-pipeline
git archive origin/estado-pipeline | tar -xf -    # mlflow_estado/, manifestos, previsões
gh release download modelos --dir models/registro  # modelos (precisa do gh autenticado)
python -m src.rastreio                             # lista os ciclos registrados no MLflow
```

Atenção: isso sobrescreve o `data/raw/manifest.json` local. Para voltar: `git checkout -- data/raw/manifest.json`.

## Limitações

- Não dá para testar o workflow inteiro fora do Actions. A lógica de recuperar e guardar o estado foi simulada localmente contra um repositório de teste, em três execuções: a primeira, a segunda recuperando o estado e uma terceira com o ciclo falhando. Os passos que usam o `gh` (Release) só rodam no Actions.
- A evidência oficial (execução pelo `schedule`, dois ciclos, a falha simulada) depende de o workflow estar na `main`.
