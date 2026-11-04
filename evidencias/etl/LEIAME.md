# Evidências do ETL (critérios 1 e 2)

Dois ciclos executados localmente em 04/11/2026 com `notebooks/01_etl.ipynb`:

| | Ciclo 1 (inicial) | Ciclo 2 (lote adicional) |
|---|---|---|
| Notebook executado | `01_executado_ciclo1.ipynb` | `01_executado_ciclo2.ipynb` |
| Diagnóstico | `diagnostico_20261104T170220Z-local.json` | `diagnostico_20261104T170233Z-local.json` |
| `data_version` | `dv018-f95c3940` | `dv019-aa1a1287` |
| Lotes | 18 (2015-07 a 2016-12) | 19 (+ `lote_2017-01`) |
| Linhas brutas | 78.703 | 82.384 |
| Linhas após o ETL | 78.590 | 82.263 |
| Removidas por regra de domínio | 113 (sem hóspedes) | 121 (sem hóspedes) |
| Duplicatas removidas | 0 (estratégia `entre_lotes`) | 0 |

O ciclo 2 identificou sozinho o lote novo (campo `lotes_novos_vs_ciclo_anterior` do diagnóstico) e mudou a `data_version`, sem reprocessar nem duplicar os 18 lotes anteriores.

Os testes `tests/test_ingestao.py` e `tests/test_preprocessamento.py` cobrem o mesmo comportamento com dados sintéticos: reentrega de lote, lote alterado, ausência de vazamento no pré-processador.

> São evidências de desenvolvimento local. As evidências oficiais do critério 7 (execução por `schedule`) virão do GitHub Actions.
