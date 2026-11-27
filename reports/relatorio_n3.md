# Pipeline MLOps para previsão de cancelamento de reservas

**Trabalho N3 · Análise Preditiva** · Católica de Santa Catarina · Prof. Rodrigo Hagstrom · novembro de 2026

**Projeto N3 · Aprendizagem de Máquina** · turma 8NA · novembro de 2026

Repositório: [github.com/GabrielKaring/projeto-n3-mlops-hotel](https://github.com/GabrielKaring/projeto-n3-mlops-hotel)

## Introdução

Um hotel vende quartos sabendo que parte das reservas vai ser cancelada, mas não sabe quais. Se aceita reservas demais, recusa hóspede no balcão; se aceita de menos, fica com quarto vazio. Este trabalho constrói uma solução de aprendizagem de máquina que estima, no momento em que a reserva entra, a probabilidade de ela ser cancelada.

**Objetivo.** Treinar e manter atualizado um classificador de cancelamento de reservas, a partir de um dataset público (*Hotel Booking Demand*), que possa ser usado com uma reserva nova e sem rótulo.

**Quem usaria.** A gestão de receita e de reservas do hotel, para decidir o overbooking de cada data e quais reservas pedem depósito ou confirmação. A equipe de dados opera o pipeline e acompanha os ciclos.

**Visão geral da solução.** Os dados chegam em lotes mensais. A cada ciclo, um workflow agendado no GitHub Actions refaz o ETL, elege as features pela votação de três métodos de seleção, faz uma regressão logística e uma Random Forest competirem, registra tudo no MLflow e só troca o modelo em uso se o novo ganhar dele por uma regra definida antes. Um monitoramento de drift avisa quando o perfil das reservas muda.

**Organização do relatório.** A seção 1 define o problema, os dados e a avaliação; as seções 2 a 7 seguem o ciclo do pipeline (ETL, seleção e votação de features, algoritmos, MLflow, workflow, promoção e previsões, com a análise de erros); a 8 mostra o uso com uma reserva nova; a 9 traz o recurso adicional de drift; a 10 discute as limitações; e a 11 reúne a conclusão e as considerações finais.

## Integrantes e divisão

Cada critério do enunciado virou uma feature com dono (`config/equipe.yaml`). O `HISTORICO_DESENVOLVIMENTO.md`, gerado a partir do git, mostra o que cada um fez.

| Feature | Critério | Dono |
|---|---|---|
| F1 a F4 | Problema e dados, ETL, três métodos de seleção, votação | Gabriel Karing |
| F5 | Dois algoritmos disputam | Gabriel Wendorff |
| F6 | MLflow e rastreabilidade | José |
| F7 e F8 | Workflow e agendamento; eleição, promoção e previsões | Diego |
| F9 | Repositório, relatório e evidências | Vinicius |
| F10 | Recurso adicional: drift com alerta | Lucas e Guilherme |

## 1. Problema, dados e avaliação

O objetivo é prever, no momento da reserva, se ela vai ser cancelada. Com isso o hotel consegue controlar overbooking, pedir depósito nas reservas de risco e planejar a ocupação.

**Quem usaria:** a gestão de receita e de reservas do hotel. Para cada reserva que entra, ela recebe a probabilidade de cancelamento e decide quanto overbooking aceitar em cada data, quais reservas exigem depósito ou confirmação e qual ocupação esperar de fato. A equipe de dados acompanha os ciclos no MLflow e o alerta de drift.

Usamos o *Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019, *Data in Brief*, licença CC BY): 119.390 reservas de dois hotéis em Portugal, com chegadas de jul/2015 a ago/2017. O alvo é `is_canceled`, com 37% de positivos.

- **Métrica principal: F1 macro.** Com 63% contra 37%, a acurácia premiaria um modelo que ignora o cancelamento. O F1 macro pesa as duas classes igual. Acompanhamos também ROC-AUC, PR-AUC, acurácia balanceada, recall e precisão do cancelamento.
- **Partição cronológica.** O teste é formado pelos últimos 20% de cada ciclo por data de chegada e não entra no treino nem na seleção: a seleção de features e a competição entre A e B usam só o desenvolvimento, com `TimeSeriesSplit` de 4 folds. O teste só é usado na comparação final vencedor × vigente e na análise de erros (seção 7, onde explicamos por que isso respeita a regra do enunciado).
- **Sem vazamento.** Tudo o que aprende com os dados (limites de outlier, escala, categorias) fica dentro do Pipeline e é ajustado só no treino. Há teste automático para isso.
- **Colunas descartadas** por só existirem depois da reserva: `reservation_status` e `reservation_status_date` (revelam o desfecho), `assigned_room_type`, `booking_changes` e `country` (o próprio artigo do dataset avisa que a nacionalidade costuma ser corrigida no check-in).

Todas essas decisões foram fixadas antes de treinar qualquer modelo e estão em `docs/etapa1_problema_dados_avaliacao.md` e `config/config.yaml`.

### Exploração dos dados

Antes de escolher tratamentos e modelos, olhamos como o cancelamento se distribui (os gráficos usam os 26 lotes depois do ETL, 119.209 reservas; a página "Os dados" da interface mostra os mesmos cortes de forma interativa).

![Taxa de cancelamento por mês de chegada, por hotel. A linha tracejada é a média (37%).](img/eda_tempo.png)

- **O cancelamento muda com o tempo e com o hotel.** O City Hotel cancela 42% das reservas, contra 28% do Resort, e a taxa sobe em 2017. Por isso a partição é cronológica (treina no passado, testa no período mais recente) e o monitoramento de drift compara cada lote com o mesmo mês do ano anterior (seção 9).
- **Classes desbalanceadas** (63% contra 37%): daí o F1 macro como métrica e o `class_weight` balanceado nos dois algoritmos.

![Taxa de cancelamento por tipo de depósito, antecedência e pedidos especiais. Vermelho = acima da média.](img/eda_atributos.png)

- **Depósito:** as reservas *Non Refund* cancelam 99% das vezes, contra 28% das sem depósito. É contraintuitivo, e é o atributo que os três métodos de seleção elegem por unanimidade em todos os ciclos (seção 3).
- **Antecedência:** a taxa cresce de 9% (até 6 dias) para 68% (um ano ou mais). Reservar muito cedo é sinal de plano ainda incerto.
- **Pedidos especiais:** quem não faz nenhum cancela 48%; com um ou mais, cerca de 20%. O pedido indica compromisso com a viagem.
- **Ausentes e duplicatas:** `company` (94%) e `agent` (14%) vazios querem dizer "sem empresa" e "sem agência", e viram `tem_empresa` e `tem_agente` em vez de ser descartados. As duplicatas são discutidas na seção 2.

## 2. ETL e chegada de dados novos

Os dados foram quebrados em lotes mensais. O ciclo inicial tem 18 lotes (jul/2015 a dez/2016), e os 8 de 2017 ficam em `data/incoming/` para simular dados chegando. Cada execução libera um lote novo.

- `data/raw/manifest.json` guarda o `sha256`, as linhas e o período de cada lote. Um lote repetido é ignorado; um lote alterado derruba o pipeline.
- `data_version = dvNNN-<hash>` identifica o conjunto de dados, e o `cycle_id` identifica a execução. Os dois acompanham o ciclo do começo ao fim.
- **Duplicatas:** 29% das linhas iniciais são idênticas a outra, e a taxa de cancelamento nelas é 63%. Na maioria são reservas de grupo canceladas juntas, um comportamento real. Por isso só removemos a linha que reaparece em outro lote (reentrega de dado). Remover todas derrubaria a taxa de 36% para 25%.
- Atributos derivados: `total_noites`, `total_hospedes`, `valor_estimado_estadia`, `razao_cancel_previo`, `tem_agente`, `tem_empresa`, `mes_chegada`, entre outros.

## 3. Seleção de features: três métodos e votação

Os três métodos rodam sobre o mesmo X e y, ajustados só no treino de cada fold:

- **SelectKBest** com informação mútua;
- **importância do Random Forest** (200 árvores);
- **RFE** com regressão logística.

Cada método vota nas suas 20 melhores (`k_top = 20`). Uma feature é eleita com **pelo menos 2 votos**, e o empate é desfeito pelo ranking médio, com piso de 10 features. A regra está no `config.yaml` desde antes da primeira execução.

Trecho da tabela de votos do 1º ciclo (dv019, 18 eleitas). A tabela completa de cada ciclo fica em `evidencias/votacao/` e no MLflow.

| Feature | SelectKBest | Random Forest | RFE | Votos | Eleita |
|---|---|---|---|---|---|
| razao_cancel_previo | 1º | 1º | 1º | 3 | sim |
| deposit_type_No Deposit | 4º | 2º | 1º | 3 | sim |
| deposit_type_Non Refund | 3º | 4º | 1º | 3 | sim |
| required_car_parking_spaces | 11º | 11º | 1º | 3 | sim |
| market_segment_Groups | 8º | 8º | 19º | 3 | sim |
| lead_time | 5º | 5º | 30º | 2 | sim |
| previous_cancellations | 2º | 3º | 43º | 2 | sim |
| agent_9 | 49º | 20º | 1º | 2 | sim |
| adr | 7º | 7º | 70º | 2 | sim |
| mes_chegada | 28º | 14º | 31º | 1 | não |
| hotel_Resort Hotel | 14º | 26º | 52º | 1 | não |

No RFE, várias features empatam em 1º porque ele só ordena as que eliminou; as que sobram até o fim ficam todas em 1º. Os três métodos concordam no núcleo (histórico de cancelamento, tipo de depósito, vaga de garagem, grupos), e a votação tira o ruído de cada um isolado.

## 4. Os dois algoritmos

Os dois candidatos usam as mesmas features eleitas, os mesmos folds e a mesma métrica. Cada um é um Pipeline completo (pré-processamento + features eleitas + algoritmo), salvo junto.

- **A:** regressão logística (`class_weight=balanced`);
- **B:** Random Forest (300 árvores, profundidade 16, `class_weight=balanced_subsample`).

| Ciclo | Lote novo | Features | A · F1 macro (CV) | B · F1 macro (CV) | Vencedor |
|---|---|---|---|---|---|
| dv019 | jan/17 | 18 | 0,747 ± 0,026 | 0,790 ± 0,037 | B |
| dv020 | fev/17 | 17 | 0,769 ± 0,045 | 0,787 ± 0,034 | B |
| dv021 | mar/17 | 20 | 0,731 ± 0,064 | 0,767 ± 0,043 | B |
| dv022 | abr/17 | 16 | 0,711 | 0,744 | B |
| dv023 | mai/17 | 17 | 0,707 | 0,758 | B |
| dv024 | jun/17 | 17 | 0,695 | 0,753 | B |
| dv025 | jul/17 | 19 | 0,695 | 0,764 | B |
| dv026 | ago/17 | 19 | 0,701 | 0,761 | B |

Médias do CV dos oito ciclos executados no GitHub Actions (`evidencias/actions/mlflow/historico_runs.csv`); o desvio entre folds está nos runs do MLflow. O B ganhou nos oito ciclos, também em ROC-AUC (0,83 a 0,85 contra 0,81 a 0,84 do A). Como as features foram eleitas olhando o desenvolvimento inteiro, o CV podia sair otimista. Para medir isso, refizemos a eleição com o ranking de cada fold isolado: a diferença ficou entre -0,03 e +0,03, sem mudar o vencedor.

## 5. MLflow e rastreabilidade

Todo run leva as tags `cycle_id`, `data_version`, `etapa`, o commit do código e, no Actions, o `github_run_id`. Cada ciclo gera 5 runs:

- `01_drift`: PSI de cada feature, variação da taxa de cancelamento e o status do alerta (seção 9);
- `02_votacao`: rankings dos três métodos, tabela de votos e features eleitas;
- `03_modelo_A` e `03_modelo_B`: parâmetros, seed, métricas de CV e o modelo serializado;
- `04_promocao`: métricas no teste do vencedor e do vigente, a regra usada, a decisão, a matriz de confusão e a análise de erros.

O run `01_drift` e a análise de erros entraram em 23/11/2026; as execuções do Actions anteriores a essa data têm os outros 4.

O histórico (SQLite + artefatos) fica em `mlflow_estado/`. Como o runner do Actions é apagado no fim, o workflow guarda essa pasta na branch `estado-pipeline` a cada execução e a recupera na seguinte. Para consultar: `python -m src.rastreio` ou `mlflow ui`. O passo a passo está no README.

## 6. Workflow único e agendamento

Um único arquivo, `.github/workflows/pipeline_mlops.yml`, com `schedule` e `workflow_dispatch`. Cada execução:

```
testes -> recupera o estado -> baixa o vigente -> libera 1 lote
  -> 01 ETL -> drift (F10) -> 02 votação -> 03 treino -> 04 promoção
  -> vencedor vai para a Release -> 05 previsões
  -> exporta o MLflow -> guarda o estado -> publica as evidências
```

- Os notebooks rodam com `nbconvert --execute`, sem tolerar erro. Se uma célula falha, o passo falha e o resto não roda.
- **O estado vive fora do runner:** a branch `estado-pipeline` guarda o MLflow, o manifesto dos lotes, o modelo vigente e as previsões. A Release `modelos` guarda o vencedor de cada ciclo, e nada lá é apagado ou substituído.
- Se o ciclo falha antes da Release, só o MLflow é gravado. O próximo ciclo parte do estado anterior e não fica vigente sem arquivo.
- `concurrency` impede dois ciclos ao mesmo tempo, porque o SQLite aceita um escritor por vez.

**Demonstração da falha:** a execução [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) usou `falhar_em = 03_treino_competicao`. O passo 03 falhou, o 04 e a Release não rodaram, e o estado gravou só o MLflow (commit "ciclo completo: false" na `estado-pipeline`). O vigente continuou o mesmo.

## 7. Promoção e previsões

A regra foi fixada antes no `config.yaml`:

- entre A e B vence a maior média de F1 macro no CV (no empate, o menor desvio);
- o vencedor e o vigente são avaliados **no teste do ciclo atual, nos mesmos registros**, e não contra um número guardado de um ciclo antigo;
- o vencedor só é promovido se passar do piso de **0,60** e superar o vigente em pelo menos **0,005**; senão o vigente fica.

Resultado nos ciclos executados no GitHub Actions (os valores são F1 macro no teste do ciclo; vencedor e vigente avaliados nos mesmos registros; "Teste" é o número de reservas do teste; entre parênteses, a diferença vencedor − vigente):

| Execução | Dados | Teste | Vencedor | Vigente | Decisão |
|---|---|---|---|---|---|
| [#2](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506) | dv019 (+ jan/2017) | 16.456 | B · 0,7310 | nenhum | **promovido** (1º ciclo) |
| [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) | falha simulada | n/a | n/a | B do #2 | parou no 03 |
| [#4](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735) | dv020 (+ fev/2017) | 17.403 | B · 0,7433 | 0,7231 | **promovido** (+0,020) |
| [#5](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641) | dv021 (+ mar/2017) | 18.392 | B · 0,7366 | 0,7415 | **mantido** (−0,005) |
| [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) (`schedule`) | dv022 (+ abr/2017) | 19.444 | B · 0,7488 | 0,7517 | **mantido** (−0,003) |
| [#7](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45262196574) (`schedule`) | dv023 (+ mai/2017) | 20.789 | B · 0,7546 | 0,7512 | **mantido** (+0,003) |
| [#8](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45474058813) (`schedule`) | dv024 (+ jun/2017) | 21.849 | B · 0,7553 | 0,7494 | **promovido** (+0,006) |
| [#9](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45679932790) (`schedule`) | dv025 (+ jul/2017) | 22.907 | B · 0,7474 | 0,7354 | **promovido** (+0,012) |
| [#10](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45878188588) (`schedule`) | dv026 (+ ago/2017) | 24.018 | B · 0,7416 | 0,7347 | **promovido** (+0,007) |

O vigente de cada ciclo foi baixado da Release e reavaliado no teste novo, por isso o valor dele muda de um ciclo para o outro. No #5 e no #6 a regra funcionou como deveria: o candidato era bom, mas não melhor que o vigente, então o vigente ficou. No #7 o candidato até ganhou, mas por 0,0034, abaixo da melhoria mínima, e o vigente também ficou. Do #8 ao #10, com os meses de 2017 mudando o perfil das reservas, o vigente antigo perdeu desempenho no teste novo e o candidato retreinado passou a regra: três promoções seguidas. Da #6 à #10 todas as execuções vieram do `schedule`, uma por dia, sem ninguém mexer. O vigente atual é o B do ciclo dv026. Os `run_id` de cada decisão estão no `models/modelo_vigente.json` da `estado-pipeline` (por exemplo `af11ab357d164c78af5d99db1ef3ef5f` para a promoção do #4 e `8834b7a20af8415f9ee2aa434a8dd42a` para a manutenção do #5).

Todo vencedor, promovido ou não, vai para a Release `modelos`. O `models/modelo_vigente.json` guarda o vigente e o histórico de decisões, com versão dos dados, features, métricas, `run_id` e motivo.

O notebook 05 carrega o vigente com o pré-processamento e prevê o **próximo lote**, que ainda não participou de nenhuma escolha. Quando esse lote chega no ciclo seguinte, o próprio notebook confere as previsões com o rótulo real.

**Uso do conjunto de teste.** O enunciado pede que o teste fique reservado à avaliação final e que as decisões recorrentes usem validação apropriada. O desenho atende a isso por quatro motivos:

- **As decisões de modelagem não veem o teste.** A eleição das features (seção 3) e a escolha entre A e B (seção 4) usam só o desenvolvimento, com validação cruzada temporal. O teste não entra no treino nem na seleção.
- **O teste é um holdout fora do tempo.** São as reservas mais recentes do ciclo, posteriores a tudo o que o candidato viu. O candidato chega ao teste já fechado, e a comparação com o vigente é a avaliação final dele.
- **A regra foi fixada antes.** Piso de 0,60 e melhoria mínima de 0,005 estão no `config.yaml` desde antes do primeiro treino; o resultado no teste não ajusta nada, só aplica a regra. A análise de erros roda depois da decisão e não a influencia.
- **A avaliação de produção é o lote futuro.** O que mede o modelo em uso de verdade é a conferência do notebook 05, com dados que não existiam no ciclo em que a previsão foi feita. Conferindo as previsões gravadas no Actions com o rótulo real dos lotes seguintes (o mesmo cálculo do notebook 05), o F1 macro foi de 0,716 em fev/2017, 0,765 em mar/2017, 0,756 em abr/2017, 0,749 em mai/2017 e 0,735 em jun/2017, na mesma faixa do teste. Em jul/2017 (0,680) e ago/2017 (0,698) ele caiu. A taxa de cancelamento de 2017 ficou acima da de 2016 desde janeiro (seção 9), e foi nesse trecho que o retreino passou a ganhar do vigente e a regra promoveu três vezes seguidas: o pipeline reagiu à mudança, mas o modelo ainda erra mais nos meses finais. A página "O modelo acertou?" da interface (seção 8) mostra essa conferência mês a mês.

**Quando os lotes acabam.** Com o `schedule` diário, o último lote de 2017 entrou em 26/11 (execução #10). Daí em diante o notebook 05 avisa que não há lote novo, mostra as últimas previsões feitas e mantém a demonstração da seção 8; os ciclos rodam sobre os mesmos dados e a decisão esperada é manter o vigente. O agendamento não foi alterado, para não mudar a simulação de chegada de dados.

### Matriz de confusão e análise de erros

O notebook 04 avalia o modelo que segue em uso (o vencedor, se promovido; senão o vigente) no teste do ciclo, com o mesmo limiar de 0,5, e grava a matriz e a análise por grupo em `evidencias/promocao/` e no run `04_promocao`. Isso só descreve o modelo; não entra na decisão.

Exemplo do último ciclo, dv026 (execução agendada [#10](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45878188588), cópia em `evidencias/actions/promocao/`): o vencedor foi promovido, então a análise é dele, o vigente atual, em 24.018 reservas do teste (22/04 a 31/08/2017).

| | Previsto: não cancela | Previsto: cancela |
|---|---|---|
| **Real: não cancelou** | 12.140 | 2.034 (falso positivo) |
| **Real: cancelou** | 3.775 (falso negativo) | 6.069 |

![Matriz de confusão do modelo em uso no teste do ciclo dv026.](img/matriz_confusao_dv026.png)

- **O erro dominante é o falso negativo:** 38,3% das reservas canceladas passaram como "vai vir", contra 14,4% de alarme falso entre as que vieram (erro total de 24,2%). Para o hotel, é quarto vazio sem aviso; o modelo é conservador ao apontar cancelamento.
- **Onde se concentra:** 99,9% dos erros são de reservas sem depósito (*No Deposit*); as *Non Refund* são todas acertadas. *Online TA* responde por 75% dos erros, com taxa de erro de 31%. Por antecedência, a faixa de 30 a 89 dias erra 32%: lá a taxa real de cancelamento é 40%, e o modelo prevê só 26%.
- **Ao longo dos 8 ciclos** (execuções locais com o mesmo código, dados e seed) o quadro se repete: *Online TA* e pelo menos uma faixa de antecedência entre 30 e 364 dias sempre aparecem entre os grupos com mais erro. Com mais dados de 2017, o falso negativo cai (de 53% no dv019 para 38% no dv026) e o falso positivo sobe (de 3% para 14%), acompanhando a taxa de cancelamento maior de 2017.

Isso aponta onde investir: um limiar ajustado ao custo de cada erro, ou atributos sobre o canal online, ajudariam mais que trocar de algoritmo.

## 8. Demonstração de uso

O uso final do modelo é responder, para uma reserva que acabou de entrar, qual a chance de ela ser cancelada. A entrada tem só o que se sabe no ato da reserva: sem o desfecho (`is_canceled`) e sem `assigned_room_type`, `booking_changes`, `reservation_status` e `reservation_status_date`. O ETL ganhou um modo inferência que aceita essa entrada e aplica exatamente as mesmas regras e atributos derivados do treino. O modelo é o vigente, carregado como no notebook 05.

```
python -m src.prever --json exemplos/reservas_exemplo.json
```

```
Reserva 1 (City Hotel, chegada 15/09/2017): chance de cancelamento 82,3% -> risco alto
Reserva 2 (City Hotel, chegada 12/10/2017): chance de cancelamento 100,0% -> risco alto
Reserva 3 (Resort Hotel, chegada 22/09/2017): chance de cancelamento 1,6% -> risco baixo
Limiar: 50% (a partir dele, risco alto; o mesmo corte da avaliação do modelo)
Modelo vigente: B do ciclo 20261126T184446Z-gha45878188588-1
```

A saída é do vigente do Actions (B do ciclo dv026), baixado da Release. As três reservas de exemplo (`exemplos/reservas_exemplo.json`) têm perfis diferentes. A primeira é de agência online no City Hotel, com 120 dias de antecedência e sem pedidos especiais; no histórico, reservas com esse perfil cancelaram 62% das vezes, contra 37% no geral. A segunda é de grupo, com depósito não reembolsável e um cancelamento anterior (no histórico, 99% das reservas *Non Refund* foram canceladas, o padrão contraintuitivo citado na seção 3). A terceira é de um hóspede que já voltou outras vezes, reservou direto, com vaga de garagem e dois pedidos especiais. A probabilidade a partir de 50% vira "risco alto", o mesmo corte usado na avaliação. A mesma demonstração roda no fim do notebook 05 em todo ciclo e fica em `evidencias/previsoes/demonstracao_<cycle_id>.csv`. Um teste automático confirma que mudar o `is_canceled` da entrada não muda a probabilidade.

Entradas inválidas são recusadas com uma mensagem clara: campo obrigatório vazio, texto num campo numérico, data que não existe (31 de setembro) e valor desconhecido em `hotel` ou `deposit_type`, os campos em que um erro de digitação mais distorceria a resposta. Nas outras categóricas, um valor que o modelo não viu no treino gera um aviso junto da previsão, porque o one-hot o trataria como categoria rara sem avisar ninguém. Os valores aceitos ficam no `config.yaml` (`inferencia`) e só valem para a demonstração; o treino não muda.

### Interface de demonstração

Para a apresentação e para quem não usa terminal, há uma interface em Streamlit, publicada em [n3-cancelamento-hotel.streamlit.app](https://n3-cancelamento-hotel.streamlit.app) (ou localmente com `streamlit run app/streamlit_app.py`, instruções no README). Ela usa as mesmas funções do pipeline e só lê: não move lotes, não treina e não grava nada versionado.

- **Testar uma reserva:** três exemplos prontos e os campos conhecidos no ato da reserva; a chance de cancelamento pelo vigente, o risco e a ação sugerida (pedir depósito ou confirmar) são recalculados a cada mudança. Um gráfico "e se mudasse a antecedência?" mostra como a probabilidade muda, o que ajuda a explicar o que o modelo aprendeu.
- **O modelo acertou?:** a conferência da seção 7, mês a mês, com matriz de confusão, previsto × real por semana e a simulação de outro corte de decisão.
- **O projeto, Os dados, Como o modelo é escolhido e Alerta de mudança:** as mesmas evidências deste relatório, lidas de `evidencias/actions/` e calculadas na hora a partir dos lotes. Cada página mostra poucos números e uma conclusão; o detalhe técnico fica recolhido.

## 9. Recurso adicional: monitoramento de drift com alerta

**Por quê:** o dataset muda de verdade com o tempo. Em abr-jun/2017 a taxa de cancelamento chega a ~43%, contra 33-37% nos meses vizinhos. Um modelo treinado no passado pode piorar sem ninguém perceber, e a promoção só olha a métrica, não o motivo.

**Como:** logo depois do ETL, cada lote novo é comparado com o mesmo mês do ano anterior. Reserva de hotel é sazonal, e comparar com o ano inteiro acusaria drift todo verão.

- PSI por feature: abaixo de 0,10 está estável, de 0,10 a 0,25 pede atenção, e a partir de 0,25 é alerta.
- Variação da taxa de cancelamento: 3 p.p. pedem atenção e 5 p.p. já são alerta.
- Lotes com menos de 500 linhas não geram alerta, porque o PSI fica ruidoso.
- O `agent` é monitorado só como informação, fora do status geral (ajuste de 23/11/2026, explicado abaixo).

No Actions, o alerta aparece como `::warning` e como tabela no resumo do job, e cada ciclo registra um run `01_drift` no MLflow com o PSI de cada feature, a variação do alvo e o status, com o mesmo `cycle_id` e `data_version` dos outros runs. A regra foi fixada antes de rodar nos dados (`docs/f10_drift.md`), com a exceção do ajuste do `agent`.

**Resultado:** no lote de jan/2017 a taxa de cancelamento subiu de 24,8% para 34,0% (+9,2 p.p.) em relação a jan/2016, e o alerta disparou. Na execução agendada [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830), o lote de abr/2017 veio com 43,5% contra 38,0% em abr/2016 (+5,5 p.p.), e o alerta apareceu como `::warning` e no resumo do job, junto com o PSI alto de `adr`, `agent` e `valor_estimado_estadia`.

**Ajuste do `agent` (23/11/2026).** Nas execuções do Actions o alerta disparou em todo lote, sempre com o `agent` entre as features em alerta. É o código da agência, com centenas de valores, e a carteira de agências se renova de um ano para o outro: contra o mesmo mês do ano anterior, o PSI dele passou de 0,25 em todos os lotes de jul/2016 a abr/2017, e em mar/2017 ele era a única feature em alerta. Agrupar em top-5 + "outros" não resolveu (o PSI ainda passa de 0,25 em 5 dos 8 lotes de 2017). Ficou a outra saída: o PSI do `agent` continua calculado, no gráfico e no MLflow, mas não entra no status geral. A mudança de canal continua no status por `tem_agente`, `market_segment` e `distribution_channel`. Como o ajuste foi feito depois de ver os dados, ele está registrado com o motivo e o teste das alternativas em `docs/f10_drift.md`. Efeito nos lotes de 2017, com os mesmos dados do Actions:

| Lote (mês) | Taxa de cancelamento (ref. → lote) | Features em alerta (no status) | Status antes | Status agora |
|---|---|---|---|---|
| jan/17 | 24,8% → 34,0% (+9,2 p.p.) | `days_in_waiting_list`, `market_segment`, `adr`, `valor_estimado_estadia` | alerta | **alerta** |
| fev/17 | 34,4% → 32,6% (−1,8 p.p.) | `previous_cancellations`, `razao_cancel_previo` | alerta | **alerta** |
| mar/17 | 30,6% → 33,8% (+3,1 p.p.) | nenhuma | alerta | atenção |
| abr/17 | 38,0% → 43,5% (+5,5 p.p.) | `adr`, `valor_estimado_estadia` | alerta | **alerta** |
| mai/17 | 35,0% → 43,8% (+8,8 p.p.) | `adr` | alerta | **alerta** |
| jun/17 | 39,6% → 43,2% (+3,6 p.p.) | `adr` | alerta | **alerta** |
| jul/17 | 32,8% → 37,4% (+4,6 p.p.) | nenhuma | alerta | atenção |
| ago/17 | 36,1% → 36,9% (+0,7 p.p.) | nenhuma | alerta | atenção |

O `agent` continua com PSI acima de 0,25 nos oito lotes, agora marcado como informativo. Os alertas que sobraram têm causa no alvo ou em outra feature, e os três lotes em que só o `agent` mudava passaram a "atenção".

![Taxa de cancelamento de cada lote contra a referência, com a regra nova, até ago/2017 (execução local). Círculo vermelho = alerta.](img/drift_historico.png)

## 10. Limitações

- **Drift de preço.** Sem o `agent`, o alerta ainda aparece em boa parte dos lotes por causa de `adr` e `valor_estimado_estadia`: as diárias sobem de um ano para o outro. É uma mudança que o modelo vê diretamente, por isso ficou no status. Um próximo passo seria comparar a diária corrigida pela inflação.
- **Momento da previsão.** O dataset não diz quando cada campo foi preenchido. Mantivemos `adults`, `children`, `stays_in_*` e `meal` supondo que refletem a reserva, mas o artigo avisa que esses campos costumam mudar até o check-in e se comportam diferente entre canceladas e não canceladas. Se forem atualizados depois, carregam parte do mesmo problema do `country` e o desempenho real seria menor.
- **Recall do cancelamento.** No teste, o B acerta 75% a 90% quando diz "vai cancelar", mas só pega de 46% a 62% dos cancelamentos, com o corte em 0,5 (seção 7). Ajustar o limiar ao custo de cada erro do hotel ficou fora do escopo.
- **Teste fixo por ciclo.** A promoção usa o teste do ciclo atual. Com teste de poucos meses, diferenças de 0,005 podem ser ruído; por isso existe a melhoria mínima, mas ela não substitui um teste estatístico.
- **Generalização.** São só dois hotéis, em um país, antes de 2018.
- **Agendamento.** O `schedule` roda todo dia às 06:00 de Brasília, mas o GitHub não garante o horário: no dia da preparação da entrega os disparos atrasaram várias horas. Por isso os ciclos #2 a #5 foram disparados à mão (`workflow_dispatch`, mesmo workflow); da #6 à #10 todos vieram do `schedule`.
- **Queda nos últimos meses.** Nas previsões conferidas com o desfecho real, o F1 macro caiu para 0,680 e 0,698 em jul e ago/2017 (seção 7). O retreino diário e a promoção reagiram, mas a regra só olha o teste do ciclo; não há um gatilho que retreine ou recalibre quando o drift alerta.
- **Fim dos lotes simulados.** São só 8 lotes de 2017. Depois do último (26/11), os ciclos agendados rodam sem dado novo e sem lote futuro para prever (seção 7).

## 11. Conclusão e considerações finais

O pipeline cumpre o ciclo completo sem ninguém mexer: chega um lote novo, o ETL é refeito, as features são eleitas de novo, os dois modelos competem, tudo é registrado no MLflow e um modelo novo só entra se ganhar do atual pelas regras definidas antes. Nos oito ciclos executados no GitHub Actions, cinco deles disparados só pelo `schedule`, a regra promoveu quando houve ganho (cinco vezes, contando o primeiro ciclo) e manteve quando não houve (três vezes, uma delas com ganho abaixo do mínimo). A Random Forest venceu sempre, com F1 macro de 0,73 a 0,76 no teste, e as previsões feitas para o mês seguinte ficaram entre 0,68 e 0,77 quando conferidas com o desfecho real. O monitoramento de drift confirmou a mudança de comportamento em 2017; depois de tirar o `agent` do status geral, o alerta deixou de disparar só por causa da troca de agências. Por fim, o modelo vigente pode ser usado com uma reserva nova, sem rótulo, pela linha de comando, pelo notebook 05 ou pela interface em Streamlit.

O principal aprendizado é que o modelo, sozinho, não resolve: o que dá confiança é o processo em volta dele (dados versionados, regra de promoção fixada antes, registro de tudo e conferência com o que aconteceu depois). Os próximos passos mais úteis para o hotel são ajustar o limiar ao custo de cada tipo de erro, já que o modelo perde muitos cancelamentos, e criar atributos sobre o canal online, que concentra três quartos dos erros.

## Evidências

| O quê | Onde |
|---|---|
| Execuções no Actions | [#2](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506), [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) (falha simulada), [#4](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735), [#5](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641), [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) a [#10](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45878188588) (agendadas) |
| Estado entre ciclos | branch `estado-pipeline` |
| Modelos de cada ciclo | Release `modelos` |
| Evidências de cada execução | artefato `evidencias-<run_id>` de cada execução |
| Tabelas de votação, competição, drift e decisões | `evidencias/` |
| Decisões e vigente | `models/modelo_vigente.json` na `estado-pipeline`; cópia em `models/modelo_vigente_actions.json` |
| Cópia das evidências do Actions | `evidencias/actions/` (votos, drift, diagnóstico, previsões e histórico do MLflow) |
| Interface de demonstração | [n3-cancelamento-hotel.streamlit.app](https://n3-cancelamento-hotel.streamlit.app), ou `streamlit run app/streamlit_app.py` (README, "Interface de demonstração") |
| Demonstração de uso | `python -m src.prever --json exemplos/reserva_exemplo.json --manifesto models/modelo_vigente_actions.json`, com o vigente baixado da Release (`models/LEIAME.md`) |
| Histórico de desenvolvimento | `HISTORICO_DESENVOLVIMENTO.md` |
| Testes | `python -m pytest -q` (219 testes) |

As tabelas das seções 3, 4 e 9 e a análise de erros da seção 7 vêm de uma execução local com o mesmo código e os mesmos dados. Como a seed é fixa, os números batem com o Actions (o F1 do 1º ciclo é 0,731032199574 nos dois).

## Referências

- ANTONIO, N.; ALMEIDA, A.; NUNES, L. Hotel booking demand datasets. *Data in Brief*, v. 22, p. 41-49, 2019. DOI: 10.1016/j.dib.2018.11.126.
- MOSTIPAK, J. *Hotel booking demand*. Kaggle. Disponível em: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825-2830, 2011.
- MLFLOW. *MLflow documentation*. Disponível em: https://mlflow.org/docs/latest/.
- SIDDIQI, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Hoboken: Wiley, 2006.
- GITHUB. *GitHub Actions documentation*. Disponível em: https://docs.github.com/actions.
