# Etapa 1 · Problema, dados e avaliação

Critérios 1 e 2 do enunciado. Tudo aqui foi definido **antes** de qualquer treino e está espelhado em `config/config.yaml`.

## 1. Problema

**Prever se uma reserva de hotel será cancelada** (classificação binária), no momento em que a reserva é feita.

Para o hotel, saber o risco de cancelamento permite gerir overbooking, exigir depósito em reservas de risco e planejar a ocupação. Os dois erros custam: tratar um hóspede confiável como risco prejudica a relação, e deixar passar um cancelamento deixa quarto vazio.

## 2. Dados

| Item | Valor |
|---|---|
| Fonte | *Hotel Booking Demand*, Antonio, Almeida & Nunes (2019), *Data in Brief* 22:41-49, DOI 10.1016/j.dib.2018.11.126. Cópia no Kaggle: `jessemostipak/hotel-booking-demand` |
| Licença | CC BY (declarada no artigo de dados original, de acesso aberto). A página do Kaggle não foi consultada para este documento. |
| Unidade de observação | Uma reserva |
| Volume | 119.390 reservas, 32 colunas, dois hotéis em Portugal (resort no Algarve e city hotel em Lisboa) |
| Janela | Chegadas de 01/07/2015 a 31/08/2017 |
| Alvo | `is_canceled` (1 = cancelada); 37% de positivos no conjunto completo |

**Limitações conhecidas**
- Não existe ID de reserva. Linhas idênticas não podem ser provadas duplicadas (ver seção 5).
- Só dois hotéis, em um país, antes de 2018. Não generaliza para outros mercados nem para o pós-pandemia.
- O dataset não informa quando cada campo foi preenchido, então o "momento da previsão" é uma suposição (seção 3).
- Em 2017 a taxa de cancelamento sobe para ~43% em abr-jun (contra 33-37% nos meses vizinhos). Há mudança de distribuição real ao longo do tempo, útil para o monitoramento de drift do critério 10.

## 3. Momento da previsão e colunas descartadas

Previsão **no ato da reserva**. Só entram atributos conhecidos nesse instante.

| Coluna | Decisão | Motivo |
|---|---|---|
| `reservation_status` | Descartada | Contém "Canceled": é o próprio alvo |
| `reservation_status_date` | Descartada | Data do desfecho |
| `assigned_room_type` | Descartada | Atribuído no check-in; em reservas canceladas quase nunca difere do reservado (1,8% contra 18,8% nas não canceladas) |
| `booking_changes` | Descartada | Acumula alterações até o desfecho |
| `arrival_date_year/month/day_of_month` | Substituídas | Viram `mes_chegada` e `dia_semana_chegada`. O ano não generaliza para datas futuras |
| `company` | Substituída | 94% ausente; vira `tem_empresa` |
| `total_of_special_requests`, `required_car_parking_spaces` | **Mantidas** | Suposição: fazem parte da reserva. Se o grupo concluir que são registradas depois, retirar em `config.yaml` |

Atributos derivados: `total_noites`, `total_hospedes`, `tem_criancas`, `valor_estimado_estadia`, `razao_cancel_previo`, `tem_agente`, `tem_empresa`, `mes_chegada`, `dia_semana_chegada`.

## 4. Métrica e partição

**Métrica principal: F1 macro** (maximizar). Com 63% de não cancelados e 37% de cancelados, a acurácia premiaria um modelo que ignora a classe minoritária. O F1 macro pesa as duas classes igualmente. Secundárias: ROC-AUC, PR-AUC, acurácia balanceada, recall e precisão do cancelamento.

**Partição cronológica** (o dado é uma série temporal; embaralhar vazaria o futuro):

1. Teste = últimos 20% dos dados do ciclo **por data de chegada**, cortando na fronteira de um dia. Fica reservado à avaliação final.
2. Desenvolvimento = o restante. Seleção de features, competição de modelos e promoção usam `TimeSeriesSplit` (4 folds, janela expansiva) dentro dele.
3. Mínimo de 5.000 linhas no desenvolvimento; abaixo disso o ciclo falha em vez de seguir com dado insuficiente.

No ciclo inicial: dev com 62.807 linhas (01/07/2015 a 23/09/2016) e teste com 15.783 (24/09/2016 a 31/12/2016).

**Sem vazamento:** o que aprende com os dados (limites de outliers, média/desvio, categorias) fica dentro de `construir_preprocessador` e é ajustado só no treino. `tests/test_preprocessamento.py` verifica isso.

## 5. Tratamento de duplicatas (decisão justificada)

Nos lotes iniciais, 22.999 linhas (29%) são idênticas a outra. Em vez de removê-las por reflexo, medimos quem são (o notebook 01 recalcula a cada ciclo):

- Taxa de cancelamento nas repetidas: **63%**, contra 25% nas demais.
- 82% das repetidas são dos segmentos *Groups* e *Offline TA/TO*; o maior grupo tem 180 linhas idênticas.

Isso aponta para **reservas de vários quartos de um mesmo pedido cancelando juntas**, comportamento real que o modelo deve aprender. Remover tudo derrubaria a taxa de cancelamento de 36% para 25%.

Estratégia padrão (`etl.duplicatas.estrategia: entre_lotes`): mantém cópias dentro do mesmo lote e remove só a linha que **reaparece em outro lote** (reentrega de dados). A opção `todas` fica disponível para análise de sensibilidade, que seria um bom item para o relatório.

## 6. Chegada de novos registros

- Os dados são divididos em **lotes mensais** (`data/raw/lote_AAAA-MM.csv.gz`). O ciclo inicial tem 18 (jul/2015 a dez/2016); os 8 seguintes (jan a ago/2017) aguardam em `data/incoming/`.
- `data/raw/manifest.json` registra cada lote incorporado com `sha256`, linhas e período.
- Registrar o mesmo conteúdo duas vezes é ignorado; alterar um lote já registrado derruba o pipeline.
- `data_version = dvNNN-<hash8>` (NNN = nº de lotes; hash dos conteúdos). `cycle_id = <timestamp UTC>-<origem>`, com origem `gha<run_id>` no GitHub Actions ou `local`.
- Para simular a chegada de dados, cada ciclo executa `python -m src.ingestao --liberar 1`.

**Demonstração (evidência do ETL):** `evidencias/etl/LEIAME.md` compara o ciclo inicial (`dv018`) com o ciclo com lote adicional (`dv019`).
