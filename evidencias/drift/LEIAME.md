# Evidências do monitoramento de drift (critério 10)

Execução local em 23/11/2026, com o código da `main` **depois** do ajuste que tirou o `agent` do status geral (`drift.informativas: [agent]`, ver [`docs/f10_drift.md`](../../docs/f10_drift.md)). Foram oito ciclos seguidos, um para cada lote de 2017 que chega de `data/incoming/`, na ordem do workflow: `python -m src.ingestao --liberar 1`, depois o notebook 01 e depois `python -m src.drift --historico`.

O drift usa o `cycle_id` e a `data_version` do `ciclo_atual.json` gravado pelo notebook 01, então cada arquivo pertence a um ciclo. Como a seed e os dados são fixos, as `data_version` batem com as do Actions. O lote de abr/2017, por exemplo, tem `dv022-b2d8eb3a` aqui e na execução agendada #6, em [`../actions/drift/`](../actions/drift/).

As evidências do Actions em `../actions/drift/` foram calculadas **antes** do ajuste, e lá todos os lotes aparecem em `alerta`. Estes arquivos mostram o resultado com a regra atual.

| Arquivo | Conteúdo |
|---|---|
| `drift_<cycle_id>.json` | Status geral, drift do alvo, PSI e status de cada feature, features informativas, referência e limiares |
| `drift_<cycle_id>.csv` / `.png` | Tabela e gráfico do PSI por feature do lote novo. A coluna `no_status` é `False` para o `agent`, que aparece em cinza no gráfico |
| `historico_20261123T141124Z-local.csv` / `.png` | Último ciclo (26 lotes): cada lote comparado com a sua referência |

## Resultado por ciclo

| Ciclo | `cycle_id` | `data_version` | Lote novo × referência | Cancelamento | Alvo | Features em alerta (no status) | Status |
|---|---|---|---|---|---|---|---|
| 1 | `20261123T140902Z-local` | `dv019-aa1a1287` | 2017-01 × 2016-01 | 24,8% → 34,0% (+9,2 p.p.) | alerta | days_in_waiting_list, market_segment, adr, valor_estimado_estadia | **alerta** |
| 2 | `20261123T140925Z-local` | `dv020-ef6bc6a5` | 2017-02 × 2016-02 | 34,4% → 32,6% (−1,8 p.p.) | ok | previous_cancellations, razao_cancel_previo | **alerta** |
| 3 | `20261123T140944Z-local` | `dv021-ccc53eec` | 2017-03 × 2016-03 | 30,6% → 33,8% (+3,1 p.p.) | atenção | nenhuma | atenção |
| 4 | `20261123T141005Z-local` | `dv022-b2d8eb3a` | 2017-04 × 2016-04 | 38,0% → 43,5% (+5,5 p.p.) | alerta | adr, valor_estimado_estadia | **alerta** |
| 5 | `20261123T141024Z-local` | `dv023-c0e59342` | 2017-05 × 2016-05 | 35,0% → 43,8% (+8,8 p.p.) | alerta | adr | **alerta** |
| 6 | `20261123T141043Z-local` | `dv024-76f1ab9c` | 2017-06 × 2016-06 | 39,6% → 43,2% (+3,6 p.p.) | atenção | adr | **alerta** |
| 7 | `20261123T141103Z-local` | `dv025-ac9d46fc` | 2017-07 × 2016-07 | 32,8% → 37,4% (+4,6 p.p.) | atenção | nenhuma | atenção |
| 8 | `20261123T141124Z-local` | `dv026-707ae968` | 2017-08 × 2016-08 | 36,1% → 36,9% (+0,7 p.p.) | ok | nenhuma | atenção |

São 5 alertas e 3 "atenção". Antes do ajuste eram 8 alertas. A tabela bate com a da seção 9 do relatório. O `agent` passa de 0,25 nos oito lotes e aparece em `features_informativas_alerta`, fora do status.

O monitoramento mostra a mudança descrita na Etapa 1. De abril a junho de 2017 o cancelamento fica perto de 43%, de 3,6 a 8,8 p.p. acima do mesmo mês de 2016.

## Histórico completo

No `historico_20261123T141124Z-local.csv` há 25 lotes com referência. Com a regra atual, 22 dão `alerta` e 3 dão `atencao` (mar, jul e ago/2017). Antes do ajuste eram 25 alertas. Fora de 2017 o ajuste não muda o status de nenhum lote:

- **2015-08 a 2016-06 (11 lotes):** ainda não existe o mesmo mês do ano anterior. A referência é a janela de lotes anteriores, que mistura estações, e todos dão alerta, quase sempre por `lead_time` ou `adr`.
- **2016-07 em diante (14 lotes):** a referência é sazonal, e o resultado é 11 alertas e 3 "atenção".

As features do status que mais passam de 0,25 são `adr` (17 lotes), `lead_time` (10), `valor_estimado_estadia` e `customer_type` (9) e `market_segment` (8). É o drift de preço descrito nas limitações do relatório (seção 10).

## MLflow

Cada ciclo gravou um run `01_drift` no MLflow local, com o mesmo `cycle_id` e `data_version` da tabela acima. A métrica `alerta` vale 1 nos cinco lotes em alerta e 0 nos três em "atenção", e `alvo_diferenca_pp` bate com a coluna de cancelamento. O banco local não é versionado. No Actions, ele fica em `mlflow_estado/`, na branch `estado-pipeline`.

## Testes

`tests/test_drift.py` cobre com dados sintéticos:

- o cálculo do PSI, a escolha da referência e o drift do alvo;
- a amostra mínima e a gravação das evidências;
- o uso do `cycle_id` do notebook 01 e as anotações do GitHub Actions;
- o `agent` como feature informativa e o registro no MLflow;
- a posição do passo no workflow: entre o notebook 01 e o 02, sem parar o ciclo no alerta e gravando no MLflow.

> É uma evidência de execução local. A execução oficial é a do Actions, em que o passo "Monitoramento de drift (F10)" de cada execução gera os mesmos arquivos, que vão no artefato `evidencias-<run_id>`.
