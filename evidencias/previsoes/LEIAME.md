# Evidências das previsões com o modelo vigente (critério 8)

Mesma execução local de `../promocao/`, em 19/11/2026: dois ciclos, com a liberação do lote `lote_2017-01` entre eles.

| Arquivo | Conteúdo |
|---|---|
| `05_executado_ciclo1.ipynb` | Notebook 05 do ciclo 1 |
| `05_executado_ciclo2.ipynb` | Notebook 05 do ciclo 2, com a conferência das previsões do ciclo 1 |
| `previsoes_<cycle_id>.csv` | Uma linha por reserva: `lote_id`, `linha_origem`, `arrival_date`, probabilidade de cancelamento, classe prevista, o ciclo e o vigente que previu |

## Onde as previsões são feitas

No **próximo lote de `data/incoming/`**: reservas que ainda não entraram no pipeline e por isso não participaram da escolha do modelo. O teste do ciclo não serve, porque já foi usado na promoção (notebook 04).

O vigente é carregado de `models/modelo_vigente.json`. O arquivo é o `Pipeline` completo da F5: pré-processamento, seleção das features eleitas e algoritmo. O lote passa pelo mesmo ETL do notebook 01. O lote simulado traz o rótulo, porque o dataset é histórico, mas ele não é usado na previsão nem gravado no CSV.

## Resultado

| Ciclo | Vigente que previu | Lote | Reservas previstas | Cancelamento previsto | Probabilidade média |
|---|---|---|---|---|---|
| 1 | B do ciclo `20261120T012057Z-local` | `lote_2017-01` | 3.673 (8 removidas pelo ETL) | 22,3% | 0,365 |
| 2 | B do ciclo `20261120T012217Z-local` | `lote_2017-02` | 4.168 (9 removidas pelo ETL) | 20,2% | 0,357 |

## Conferência quando o lote chega

No ciclo 2, o `lote_2017-01` já tinha entrado no pipeline com o rótulo real, e as previsões feitas no ciclo 1 foram conferidas:

| Previsões de | Lote | n | f1_macro | ROC-AUC | PR-AUC | Recall | Precisão |
|---|---|---|---|---|---|---|---|
| ciclo 1 | `lote_2017-01` | 3.673 | 0,7514 | 0,8580 | 0,7977 | 0,5324 | 0,8110 |

## Limitações

- No último ciclo não sobra lote em `data/incoming/`. O notebook avisa e termina sem erro.
- A conferência depende de as previsões anteriores estarem disponíveis. No Actions, `evidencias/previsoes/` precisa ser persistida entre execuções (F7).
- Execução local. A evidência oficial deve vir de ciclos agendados no Actions (gate GF).
