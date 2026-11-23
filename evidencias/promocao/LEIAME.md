# Evidências da eleição e da promoção (critério 8)

Execução local em 19/11/2026: notebooks 01 → 05, liberação do lote `lote_2017-01` (`python -m src.ingestao --liberar 1`) e de novo notebooks 01 → 05.

| Arquivo | Conteúdo |
|---|---|
| `04_executado_ciclo1.ipynb` | Notebook 04 do ciclo 1 (primeiro ciclo, sem vigente) |
| `04_executado_ciclo2.ipynb` | Notebook 04 do ciclo 2 (vencedor contra o vigente do ciclo 1) |
| `decisao_<cycle_id>.json` | Decisão de cada ciclo: regra, métricas do vencedor e do vigente no teste, motivo, features, `run_id` |
| `erros_<cycle_id>.json`, `erros_por_grupo_<cycle_id>.csv`, `matriz_confusao_<cycle_id>.png` | Matriz de confusão e análise de erros do modelo em uso, no teste do ciclo (gerados a partir de 23/11/2026; não existiam nesta execução de 19/11) |

## Regra

Fixada antes de rodar, em `config.yaml` (`promocao`): `f1_macro`, maximizar, melhoria mínima de **0,005**, piso de **0,60**.

1. Entre A e B vence a maior média de `f1_macro` no CV do dev (números da F5). Empate: menor desvio entre os folds.
2. O vencedor e o vigente são avaliados no teste do ciclo, os mesmos registros para os dois.
3. Sem vigente, o vencedor é promovido se passar no piso. Com vigente, só é promovido se passar no piso e superar o vigente em pelo menos 0,005.

## Resultado

| Ciclo | `cycle_id` | `data_version` | A × B (CV) | Teste | Vencedor | Vigente | Decisão |
|---|---|---|---|---|---|---|---|
| 1 | `20261120T012057Z-local` | `dv018-f95c3940` | 0,7406 × **0,7783** | 15.783 linhas, 24/09 a 31/12/2016 | B: 0,6773 | nenhum | **promovido**: primeiro ciclo, passa no piso |
| 2 | `20261120T012217Z-local` | `dv019-aa1a1287` | 0,7471 × **0,7904** | 16.456 linhas, 10/10/2016 a 31/01/2017 | B: 0,7310 | B do ciclo 1: 0,6877 | **promovido**: +0,0433 |

Os valores de vencedor e vigente são `f1_macro` no teste do ciclo.

| Ciclo | Run `04_promocao` | Run de treino do vencedor (`03_modelo_B`) | Arquivo do vencedor |
|---|---|---|---|
| 1 | `18b69869ea874379b362a3f484c63bf9` | `6b0c7dd65ec044e19c74c79b674d7d0d` | `modelo_20261120T012057Z-local_B.joblib` (19,0 MB) |
| 2 | `bdbda4d2b03a4197b1f4cb8a74c4ba89` | `2ce3d20779a448838a973cffa3e5fe3c` | `modelo_20261120T012217Z-local_B.joblib` (19,7 MB) |

- O vencedor de cada ciclo é guardado comprimido em `models/registro/` (fora do git). Nenhum modelo é apagado. No Actions, o workflow (F7) envia esses arquivos para a Release do GitHub.
- `models/modelo_vigente.json` acumula as duas decisões no histórico, e o vigente final é o B do ciclo 2.
- O caso "mantido" (melhora insuficiente, piora ou abaixo do piso) não aconteceu nestes dois ciclos. Ele está coberto em `tests/test_promocao.py`.

## Limitações

- Os dois ciclos rodaram localmente. As decisões dos ciclos do Actions, inclusive o agendado, estão no histórico de `models/modelo_vigente_actions.json`.
- O teste do ciclo decide a promoção, então as previsões do notebook 05 são feitas no próximo lote de `data/incoming/`, que não participou da escolha. Ver `../previsoes/`.
- O vigente é avaliado como foi salvo, sem retreino: no ciclo 2 ele tinha visto dados só até 23/09/2016, e o vencedor até 09/10/2016. Parte da vantagem do vencedor vem de ter visto dados mais recentes, o que é justamente o que a promoção deve premiar.
