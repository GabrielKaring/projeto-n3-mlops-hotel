# Evidências da votação de features (critério 4)

Execução local em 05/11/2026 de `notebooks/02_votacao_features.ipynb`, seção F4. O notebook executado está em `../selecao/02_executado_ciclo1.ipynb`.

| | |
|---|---|
| Tabela de votos | `votos_20261105T123115Z-local.csv` |
| `data_version` | `dv018-f95c3940` |
| Regra (em `config.yaml`) | Top-20 por método; eleita com pelo menos 2 de 3 votos; mínimo de 10 features; desempate por ranking médio |
| Resultado | 17 features eleitas de 76 |

A regra já estava em `config.yaml` antes de o código da F3 existir.

| Votos | Features |
|---|---|
| 3 | 4 |
| 2 | 13 |
| 1 | 23 |
| 0 | 36 |

**Conjunto eleito:** `razao_cancel_previo`, `deposit_type_No Deposit`, `deposit_type_Non Refund`, `required_car_parking_spaces`, `previous_cancellations`, `market_segment_Groups`, `lead_time`, `total_of_special_requests`, `hotel_City Hotel`, `agent_1`, `distribution_channel_Direct`, `arrival_date_week_number`, `valor_estimado_estadia`, `total_noites`, `adr`, `customer_type_Transient` e `stays_in_week_nights`.

O piso de 10 features não precisou ser usado, e todas as eleitas existem nos 4 folds.

## Limitações

- O registro no MLflow é responsabilidade da F6 e será adicionado por ela ao fim do notebook 02.
- Há redundância no conjunto: `deposit_type_No Deposit` e `deposit_type_Non Refund` são quase complementares, e `total_noites` contém `stays_in_week_nights`. A votação não remove atributos redundantes.
- `deposit_type_Non Refund` tem 99% de cancelamento no dev, um padrão contraintuitivo para reservas sem reembolso. Vale citá-lo no relatório como ponto a investigar.
- O conjunto vem da média entre folds, então o CV da F5 tende a sair levemente otimista.
