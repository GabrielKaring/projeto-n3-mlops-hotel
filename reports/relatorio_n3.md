# Pipeline MLOps para previsão de cancelamento de reservas

**Trabalho N3 · Análise Preditiva** · Católica de Santa Catarina · Prof. Rodrigo Hagstrom · novembro de 2026

Repositório: [github.com/GabrielKaring/projeto-n3-mlops-hotel](https://github.com/GabrielKaring/projeto-n3-mlops-hotel)

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

Usamos o *Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019, *Data in Brief*, licença CC BY): 119.390 reservas de dois hotéis em Portugal, com chegadas de jul/2015 a ago/2017. O alvo é `is_canceled`, com 37% de positivos.

- **Métrica principal: F1 macro.** Com 63% contra 37%, a acurácia premiaria um modelo que ignora o cancelamento. O F1 macro pesa as duas classes igual. Acompanhamos também ROC-AUC, PR-AUC, acurácia balanceada, recall e precisão do cancelamento.
- **Partição cronológica.** O teste é formado pelos últimos 20% de cada ciclo por data de chegada e só é usado na promoção. No desenvolvimento, a seleção de features e a competição usam `TimeSeriesSplit` com 4 folds.
- **Sem vazamento.** Tudo o que aprende com os dados (limites de outlier, escala, categorias) fica dentro do Pipeline e é ajustado só no treino. Há teste automático para isso.
- **Colunas descartadas** por só existirem depois da reserva: `reservation_status`, `assigned_room_type`, `booking_changes` e `country` (o próprio artigo do dataset avisa que a nacionalidade costuma ser corrigida no check-in).

Todas essas decisões foram fixadas antes de treinar qualquer modelo e estão em `docs/etapa1_problema_dados_avaliacao.md` e `config/config.yaml`.

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
| dv019 | jan/2017 | 18 | 0,747 ± 0,026 | 0,790 ± 0,037 | B |
| dv020 | fev/2017 | 17 | 0,769 ± 0,045 | 0,787 ± 0,034 | B |
| dv021 | mar/2017 | 20 | 0,731 ± 0,064 | 0,767 ± 0,043 | B |

O B ganhou nos três ciclos, também em ROC-AUC (0,85 contra 0,83). Como as features foram eleitas olhando o desenvolvimento inteiro, o CV podia sair otimista. Para medir isso, refizemos a eleição com o ranking de cada fold isolado: a diferença ficou entre -0,03 e +0,03, sem mudar o vencedor.

## 5. MLflow e rastreabilidade

Todo run leva as tags `cycle_id`, `data_version`, `etapa`, o commit do código e, no Actions, o `github_run_id`. Cada ciclo gera 4 runs:

- `02_votacao`: rankings dos três métodos, tabela de votos e features eleitas;
- `03_modelo_A` e `03_modelo_B`: parâmetros, seed, métricas de CV e o modelo serializado;
- `04_promocao`: métricas no teste do vencedor e do vigente, a regra usada e a decisão.

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

Resultado nos ciclos executados no GitHub Actions (os valores são F1 macro no teste do ciclo; vencedor e vigente avaliados nos mesmos registros):

| Execução | Dados | Teste (registros) | Vencedor | Vigente | Decisão |
|---|---|---|---|---|---|
| [#2](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506) | dv019 (+ jan/2017) | 16.456 | B · 0,7310 | nenhum | **promovido** (1º ciclo, passa no piso) |
| [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) | falha simulada | n/a | n/a | B do #2 | nada promovido; fluxo parou no 03 |
| [#4](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735) | dv020 (+ fev/2017) | 17.403 | B · 0,7433 | 0,7231 | **promovido** (+0,020) |
| [#5](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641) | dv021 (+ mar/2017) | 18.392 | B · 0,7366 | 0,7415 | **mantido** (não supera o vigente) |
| [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) (`schedule`) | dv022 (+ abr/2017) | 19.444 | B · 0,7488 | 0,7517 | **mantido** (não supera o vigente) |

O vigente de cada ciclo foi baixado da Release e reavaliado no teste novo, por isso o valor dele muda de um ciclo para o outro. No #5 e no #6 a regra funcionou como deveria: o candidato era bom, mas não melhor que o vigente, então o vigente ficou. A #6 foi disparada pelo `schedule`, sem ninguém mexer. Os `run_id` de cada decisão estão no `models/modelo_vigente.json` da `estado-pipeline` (por exemplo `af11ab357d164c78af5d99db1ef3ef5f` para a promoção do #4 e `8834b7a20af8415f9ee2aa434a8dd42a` para a manutenção do #5).

Todo vencedor, promovido ou não, vai para a Release `modelos`. O `models/modelo_vigente.json` guarda o vigente e o histórico de decisões, com versão dos dados, features, métricas, `run_id` e motivo.

O notebook 05 carrega o vigente com o pré-processamento e prevê o **próximo lote**, que ainda não participou de nenhuma escolha. Quando esse lote chega no ciclo seguinte, o próprio notebook confere as previsões com o rótulo real.

## 8. Recurso adicional: monitoramento de drift com alerta

**Por quê:** o dataset muda de verdade com o tempo. Em abr-jun/2017 a taxa de cancelamento chega a ~43%, contra 33-37% nos meses vizinhos. Um modelo treinado no passado pode piorar sem ninguém perceber, e a promoção só olha a métrica, não o motivo.

**Como:** logo depois do ETL, cada lote novo é comparado com o mesmo mês do ano anterior. Reserva de hotel é sazonal, e comparar com o ano inteiro acusaria drift todo verão.

- PSI por feature: abaixo de 0,10 está estável, de 0,10 a 0,25 pede atenção, e a partir de 0,25 é alerta.
- Variação da taxa de cancelamento: 3 p.p. pedem atenção e 5 p.p. já são alerta.
- Lotes com menos de 500 linhas não geram alerta, porque o PSI fica ruidoso.

No Actions, o alerta aparece como `::warning` e como tabela no resumo do job. A regra foi fixada antes de rodar nos dados (`docs/f10_drift.md`).

**Resultado:** no lote de jan/2017 a taxa de cancelamento subiu de 24,8% para 34,0% (+9,2 p.p.) em relação a jan/2016, e o alerta disparou. Na execução agendada [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830), o lote de abr/2017 veio com 43,5% contra 38,0% em abr/2016 (+5,5 p.p.), e o alerta apareceu como `::warning` e no resumo do job, junto com o PSI alto de `adr`, `agent` e `valor_estimado_estadia`.

![Taxa de cancelamento de cada lote contra a referência. Círculo vermelho = alerta.](img/drift_historico.png)

## 9. Limitações

- **O alerta de drift dispara em quase todo lote.** O culpado é o `agent`: os IDs de agência mudam muito de um ano para o outro, e o PSI dele passa de 0,25 quase sempre. O mecanismo funciona, mas do jeito que está gera alarme demais. O próximo passo seria agrupar as agências ou tirar o `agent` do status geral.
- **Momento da previsão.** O dataset não diz quando cada campo foi preenchido. Mantivemos `adults`, `children`, `stays_in_*` e `meal` supondo que refletem a reserva, mas o artigo avisa que esses campos costumam mudar até o check-in e se comportam diferente entre canceladas e não canceladas. Se forem atualizados depois, carregam parte do mesmo problema do `country` e o desempenho real seria menor.
- **Recall do cancelamento.** No teste, o B acerta 90% quando diz "vai cancelar", mas só pega 46% dos cancelamentos, com o corte em 0,5. Ajustar o limiar ao custo de cada erro do hotel ficou fora do escopo.
- **Teste fixo por ciclo.** A promoção usa o teste do ciclo atual. Com teste de poucos meses, diferenças de 0,005 podem ser ruído; por isso existe a melhoria mínima, mas ela não substitui um teste estatístico.
- **Generalização.** São só dois hotéis, em um país, antes de 2018.
- **Agendamento.** O `schedule` roda todo dia às 06:00 de Brasília, mas o GitHub não garante o horário: no dia da preparação da entrega os disparos atrasaram várias horas. Por isso os ciclos #2 a #5 foram disparados à mão (`workflow_dispatch`, mesmo workflow), e a #6 já veio do `schedule`.

## 10. Conclusão

O pipeline cumpre o ciclo completo sem ninguém mexer: chega um lote novo, o ETL é refeito, as features são eleitas de novo, os dois modelos competem, tudo é registrado no MLflow e um modelo novo só entra se ganhar do atual pelas regras definidas antes. Nos ciclos executados, a regra promoveu quando houve ganho e manteve quando não houve. A Random Forest venceu sempre, com F1 macro de 0,73 a 0,75 no teste. O monitoramento de drift confirmou a mudança de comportamento em 2017, mas precisa de ajuste fino para não alarmar à toa.

## Evidências

| O quê | Onde |
|---|---|
| Execuções no Actions | [#2](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506), [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) (falha simulada), [#4](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735), [#5](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641), [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) (agendada) e as agendadas seguintes |
| Estado entre ciclos | branch `estado-pipeline` |
| Modelos de cada ciclo | Release `modelos` |
| Evidências de cada execução | artefato `evidencias-<run_id>` de cada execução |
| Tabelas de votação, competição, drift e decisões | `evidencias/` |
| Decisões e vigente | `models/modelo_vigente.json` na `estado-pipeline` |
| Histórico de desenvolvimento | `HISTORICO_DESENVOLVIMENTO.md` |
| Testes | `python -m pytest -q` (167 testes) |

As tabelas das seções 3, 4 e 8 vêm de uma execução local com o mesmo código e os mesmos dados. Como a seed é fixa, os números batem com o Actions (o F1 do 1º ciclo é 0,731032199574 nos dois).
