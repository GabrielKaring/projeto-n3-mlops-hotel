# Evidências das execuções no GitHub Actions

Cópia, na `main`, do que o workflow produziu no Actions. O original continua onde o workflow grava: a branch `estado-pipeline` (estado entre ciclos) e o artefato `evidencias-<run_id>` de cada execução. A cópia foi feita com `python scripts/copiar_evidencias_actions.py`, que só lê o GitHub. Para atualizar depois de novas execuções agendadas, rode o script de novo.

Ela fica numa pasta própria para não se misturar com os ciclos locais: o notebook 01 e o notebook 05 leem `evidencias/etl/` e `evidencias/previsoes/`, e não esta pasta.

## Execuções copiadas

As 8 execuções que decidiram um ciclo (dv019 a dv026; as 5 últimas disparadas pelo `schedule`), com vencedor, vigente e decisão de cada uma, estão na tabela de "Principais resultados" do [README](../../README.md#principais-resultados). A execução [45091260105](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) foi a falha simulada (`falhar_em = 03_treino_competicao`): o fluxo parou no notebook 03 e nada foi decidido, por isso ela não tem arquivos aqui.

## Arquivos

| Pasta | Conteúdo | Origem |
|---|---|---|
| `../../models/modelo_vigente_actions.json` | Manifesto do vigente e histórico das 8 decisões, com `run_id`, features, métricas e motivo | `models/modelo_vigente.json` da `estado-pipeline` |
| `votacao/` | Tabela de votos de cada ciclo | artefato de cada execução |
| `mlflow/historico_runs.csv` | Todos os runs do MLflow, com `cycle_id`, `data_version`, etapa e métricas (inclusive o CV de A e B de cada ciclo) | artefato da última execução |
| `drift/` | Drift do lote da última execução (JSON e gráfico) e o histórico de todos os lotes | artefato da última execução |
| `etl/` | Diagnóstico de antes e depois do ETL da última execução | `estado-pipeline` |
| `previsoes/` | Previsões do próximo lote feitas em cada ciclo, comprimidas (o pandas lê direto com `pd.read_csv`) | `estado-pipeline` |

Os modelos de cada ciclo estão na Release `modelos`. Para não inchar a `main`, fica de fora o que se repete: as decisões (estão no manifesto), os resultados da competição (estão no `historico_runs.csv`) e o drift e o diagnóstico das execuções anteriores (o histórico de drift já cobre todos os lotes). O conjunto completo continua no artefato `evidencias-<run_id>` de cada execução.

## Observações

- O drift copiado é o da última execução (dv026), já com o ajuste de 23/11/2026 que tirou o `agent` do status geral (`docs/f10_drift.md`): o PSI dele aparece na coluna `informativas_alerta` do histórico. As execuções até a 45154548830 rodaram antes do ajuste e marcavam todo lote como `alerta`.
- A matriz de confusão e a análise de erros do notebook 04, e a demonstração com reservas novas do notebook 05, entraram em 23/11/2026. A análise de erros da última execução está em `promocao/`, e as demonstrações das execuções 45262196574 em diante estão em `previsoes/`.
