# Evidências da competição entre modelos (critério 5)

Execução local em 19/11/2026 de `notebooks/03_treino_competicao.ipynb`, logo depois dos notebooks 01 e 02 no mesmo ciclo.

| | |
|---|---|
| Notebook executado | `03_executado_ciclo1.ipynb` |
| Métricas de A e B (média, desvio e por fold) | `resultados_20261119T175604Z-local.json` |
| `data_version` | `dv018-f95c3940` (os mesmos dados de `../etl/` e `../selecao/`) |
| Dev | 62.807 linhas, 4 folds cronológicos, 17 features eleitas |
| Modelos | `models/candidatos/20261119T175604Z-local/` (fora do git; o B tem 51 MB) |

## Resultado

`f1_macro`, média ± desvio nos 4 folds:

| Modelo | Algoritmo | f1_macro | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| A | Regressão logística | 0,7406 ± 0,0133 | 0,8275 | 0,7897 |
| B | Random forest | 0,7786 ± 0,0497 | 0,8483 | 0,8116 |

Por fold:

| Fold | A | B | B − A |
|---|---|---|---|
| 1 | 0,7380 | 0,8438 | +0,1057 |
| 2 | 0,7268 | 0,7423 | +0,0155 |
| 3 | 0,7587 | 0,7909 | +0,0322 |
| 4 | 0,7389 | 0,7373 | −0,0016 |

- O B vence em 3 dos 4 folds e tem a média maior, mas varia bem mais entre folds. No último fold, o mais próximo do teste, os dois empatam.
- Nenhuma feature eleita ficou ausente nos folds deste ciclo. Se acontecer, a coluna entra zerada e aparece em `features_ausentes` no json.
- Quem escolhe o vencedor e decide a promoção é a F8, no notebook 04, com o teste.

## O CV está otimista?

As eleitas vieram da média dos rankings dos 4 folds, então o conjunto já viu parte das validações. Para medir isso, a votação foi refeita com o ranking de cada fold isolado, e cada fold foi avaliado com o conjunto eleito só pelo seu próprio treino.

| Modelo | CV oficial | CV com eleição por fold | Diferença |
|---|---|---|---|
| A | 0,7406 | 0,7759 | −0,0353 |
| B | 0,7786 | 0,7848 | −0,0063 |

O CV oficial saiu **abaixo**, e não acima. Neste ciclo, o reaproveitamento das validações não inflou o resultado. O conjunto de cada fold se adapta melhor ao próprio período (no fold 1 entram `adults` e `total_hospedes`, por exemplo), e a regressão logística, com menos flexibilidade, é a que mais sente essa diferença.
