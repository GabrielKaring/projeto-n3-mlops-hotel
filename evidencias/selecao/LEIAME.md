# Evidências da seleção de features (critério 3)

Execução local em 05/11/2026 de `notebooks/02_votacao_features.ipynb`, seção F3, com o conjunto de desenvolvimento do ciclo inicial, já sem a coluna `country`.

| | |
|---|---|
| Notebook executado | `02_executado_ciclo1.ipynb` |
| Scores e posições por método | `rankings_20261105T123115Z-local.csv` |
| Scores de cada fold | `rankings_por_fold_20261105T123115Z-local.csv` |
| `data_version` | `dv018-f95c3940` (os mesmos dados do ciclo 1 de `evidencias/etl/`) |
| Dev | 62.807 linhas, 4 folds cronológicos, 76 features após o pré-processador (união dos folds) |

O notebook termina em erro na célula da F4 (`NotImplementedError`), que ainda não foi implementada. A seção F3 rodou inteira antes dela.

## Resultado

Features em comum no top-20:

| | SelectKBest | Random Forest | RFE |
|---|---|---|---|
| SelectKBest | 20 | 14 | 6 |
| Random Forest | 14 | 20 | 5 |
| RFE | 6 | 5 | 21 |

Estabilidade do top-20 entre folds seguidos (Jaccard):

| | Folds 1-2 | Folds 2-3 | Folds 3-4 |
|---|---|---|---|
| SelectKBest | 0,74 | 0,82 | 0,82 |
| Random Forest | 0,82 | 0,82 | 0,90 |
| RFE | 0,33 | 0,43 | 0,74 |

- Quatro features estão no top-20 dos três métodos: `deposit_type_No Deposit`, `deposit_type_Non Refund`, `razao_cancel_previo` e `required_car_parking_spaces`.
- SelectKBest e Random Forest concordam bastante entre si. O RFE traz um ponto de vista diferente, mas é o menos estável.
- Features empatadas na fronteira entram todas no top-k, e o desempate fica para a F4. Por isso o top-20 do RFE tem 21 features: `agent_250` e `agent_28` empatam na posição 20.
- `agent_28` não existe no fold 1, porque não passa de 1% naquele treino. O RFE a manteve nos outros três folds. Se for eleita, a F5 precisa tratar os ajustes em que a coluna não for gerada.

## Retirada de `country`

Na execução anterior, com a coluna, `country_PRT` estava no top-8 dos três métodos. A taxa de cancelamento no dev era de 55,6% quando `country` era PRT e de 17,4% nos demais; 74,5% das canceladas eram PRT, contra 33,0% das não canceladas.

O artigo do dataset (Antonio et al., 2019, p. 45) diz: *"It is also common for hotels not to know the correct nationality of the customer until the moment of check-in"*, e aponta que a distribuição de `Country` difere entre reservas canceladas e não canceladas. Reserva cancelada não passa pelo check-in, então a coluna pode carregar informação do desfecho, o que conflita com a previsão no ato da reserva (Etapa 1, seção 3). O artigo não informa o valor registrado antes do check-in; a concentração em PRT é compatível com vazamento, mas não o prova. A coluna foi retirada por precaução, antes do treino de qualquer modelo.

O mesmo trecho do artigo cita `Adults`, `Children`, `StaysInWeekendNights`, `StaysInWeekNights` e `Meal` como variáveis com distribuição diferente entre canceladas e não canceladas. Elas foram mantidas como suposição, registrada na Etapa 1.

## Limitação para a F5

Os scores são a média dos treinos dos folds. Um único conjunto eleito a partir dessa média viu a validação dos folds iniciais, então o CV da F5 tende a sair levemente otimista. Só a validação do último fold e o teste ficam intocados.
