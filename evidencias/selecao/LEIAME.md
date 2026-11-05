# Evidências da seleção de features (critério 3)

Execução local em 05/11/2026 de `notebooks/02_votacao_features.ipynb`, seção F3, com o conjunto de desenvolvimento do ciclo inicial.

| | |
|---|---|
| Notebook executado | `02_executado_ciclo1.ipynb` |
| Scores e posições por método | `rankings_20261105T113604Z-local.csv` |
| Scores de cada fold | `rankings_por_fold_20261105T113604Z-local.csv` |
| `data_version` | `dv018-f95c3940` (os mesmos dados do ciclo 1 de `evidencias/etl/`) |
| Dev | 62.807 linhas, 4 folds cronológicos, 89 features após o pré-processador (união dos folds) |

O notebook termina em erro na célula da F4 (`NotImplementedError`), que ainda não foi implementada. A seção F3 rodou inteira antes dela. Repetir o cálculo com a mesma seed deu scores idênticos.

## Resultado

Features em comum no top-20:

| | SelectKBest | Random Forest | RFE |
|---|---|---|---|
| SelectKBest | 20 | 15 | 7 |
| Random Forest | 15 | 20 | 8 |
| RFE | 7 | 8 | 20 |

Estabilidade do top-20 entre folds seguidos (Jaccard):

| | Folds 1-2 | Folds 2-3 | Folds 3-4 |
|---|---|---|---|
| SelectKBest | 0,82 | 0,82 | 1,00 |
| Random Forest | 0,82 | 0,82 | 1,00 |
| RFE | 0,38 | 0,54 | 0,67 |

- SelectKBest e Random Forest concordam bastante entre si. O RFE traz um ponto de vista diferente, mas é o menos estável.
- Nenhuma feature do top-20 de qualquer método deixou de existir em algum fold.

## Risco de vazamento em `country`

`country_PRT` está no top-8 dos três métodos. Na mesma partição, a taxa de cancelamento é de 55,6% quando `country` é PRT e de 17,4% nos demais; 74,5% das canceladas são PRT, contra 33,0% das não canceladas.

O artigo do dataset (Antonio et al., 2019, p. 45) diz: *"It is also common for hotels not to know the correct nationality of the customer until the moment of check-in"*, e aponta que a distribuição de `Country` difere entre reservas canceladas e não canceladas. Reserva cancelada não passa pelo check-in, então a coluna pode carregar informação do desfecho, o que conflita com a previsão no ato da reserva (Etapa 1, seção 3). O artigo não informa o valor registrado antes do check-in; a concentração em PRT é compatível com vazamento, mas não o prova.

A coluna foi mantida nesta execução: mudar a lista de features é uma decisão de F1/F2.

## Limitação para a F5

Os scores são a média dos treinos dos folds. Um único conjunto eleito a partir dessa média viu a validação dos folds iniciais, então o CV da F5 tende a sair levemente otimista. Só a validação do último fold e o teste ficam intocados.
