# Gates de entrega do Trabalho N3

Um **gate** é o conjunto de condições verificáveis que uma feature precisa cumprir para ser considerada pronta. As condições vêm do enunciado e da apresentação do professor. Os detalhes de cada feature estão em [`divisao_de_tarefas.md`](divisao_de_tarefas.md), e os donos em [`config/equipe.yaml`](../config/equipe.yaml).

**Como usar**
- O dono marca uma condição só com a evidência em mãos: arquivo, log ou `run_id`. Código que existe, mas não roda, não conta.
- O gate só passa com todas as condições marcadas.
- Print complementa, mas não substitui código e logs verificáveis.

**Pontuação de cada critério (enunciado, seção 8):** 1,0 = implementado e demonstrado; 0,5 = parcial, com evidência insuficiente ou falha relevante; 0,0 = ausente, não executável ou não comprovado.

## Painel

Situação conferida em 05/11/2026, na branch `f3-selecao` (PR #1 ainda aberto).

| Gate | O que cobre | Dono | Depende de | Situação | O que falta |
|---|---|---|---|---|---|
| G0 | Regras do grupo e do repositório | Todos | | 🔴 bloqueado | Grupo acima do limite e acesso ao repositório |
| G1 | Problema, dados e avaliação | Gabriel Karing | | ✅ passou | |
| G2 | ETL e atributos | Gabriel Karing | G1 | ✅ passou | Refazer os dois ciclos no pipeline final |
| G3 | Três métodos de seleção | Gabriel Karing | G2 | 🟡 parcial | Merge |
| G4 | Votação | Gabriel Karing | G3 | 🟡 parcial | Merge |
| G5 | Dois algoritmos | Gabriel Wendorff | G4 | ⬜ não iniciado | Tudo |
| G6 | MLflow, incluindo o registro de seleção e votação | José | G3, G4 | 🟡 parcial | Persistência no Actions (com F7), runs de F5 e F8, captura da interface |
| G7 | Workflow e agendamento | Diego | G6 | 🟡 parcial | Merge na `main`, permissão de escrita e execuções reais (incluindo pelo `schedule`) |
| G8 | Promoção e previsões | Diego | G5, G6 | ⬜ não iniciado | Tudo |
| G9 | Repositório, relatório e evidências | Vinicius | todos | 🟡 parcial | Relatório e evidências oficiais |
| G10 | Recurso adicional | Lucas e Guilherme | G2, G7 | ⬜ não iniciado | Tudo |
| GF | Teste integrado final | Todos | todos | ⬜ não iniciado | Tudo |

Ordem do trabalho: `G1 → G2 → G3 → G4 → G5 → G8 → G9 → GF`. G6 e G7 correm em paralelo desde o início, e G10 entra no fluxo pelo G7.

## G0 · Regras do grupo e do repositório

- [ ] Grupo com até 5 integrantes, ou exceção confirmada pelo professor. **Hoje são 7.**
- [ ] Repositório acessível ao professor. **Hoje é privado.**
- [ ] Todos os integrantes com acesso de escrita. **Hoje só há 2 colaboradores.**
- [ ] Commits de cada pessoa com o nome de `config/equipe.yaml`, ou com linha no `.mailmap`.
- [x] Sem tokens, senhas, ambientes virtuais ou dados restritos versionados.
- [x] Dados públicos, com fonte e licença indicadas.

## G1 · Problema, dados e avaliação

Evidência: `docs/etapa1_problema_dados_avaliacao.md`.

- [x] Tipo de problema escolhido e justificado.
- [x] Fonte, licença, unidade de observação, alvo, atributos e limitações descritos.
- [x] Métrica principal e direção definidas antes de treinar qualquer modelo.
- [x] Partição respeita a ordem cronológica, com teste reservado à avaliação final.
- [x] Forma de chegada de novos registros explicada.

## G2 · ETL e engenharia de atributos

Evidência: `evidencias/etl/`.

- [x] Ausentes, duplicatas, inconsistências e outliers verificados no notebook 01.
- [x] Tratamentos justificados e diagnóstico de antes e depois.
- [x] Transformações ajustadas só no treino, via `Pipeline`/`ColumnTransformer`.
- [x] Novos lotes identificados e incorporados sem duplicação, com `data_version`.
- [ ] Dois ciclos refeitos no pipeline final. Os atuais são locais e anteriores à retirada de `country`.

## G3 · Três métodos de seleção

Evidência: `evidencias/selecao/`.

- [x] Três métodos no notebook 02, sobre os mesmos dados.
- [x] Ranking ou importância de cada método salvo.
- [x] Ajuste só no treino e dentro de cada fold, com teste de ausência de vazamento.
- [x] Cada método entrega a lista de features aprovadas (`listas_aprovadas` e colunas `aprovada_*`).
- [ ] PR #1 integrado à `main`.

## G4 · Votação e eleição

Evidência: `evidencias/votacao/`.

- [x] Regra definida antes da comparação, em `config.yaml`, e justificada no notebook 02.
- [x] Top-k, regra de desempate e quantidade mínima de features aplicados.
- [x] O empate na fronteira do top-k está tratado: as empatadas recebem o voto.
- [x] Tabela de votos por feature gerada a cada ciclo.
- [x] Um único conjunto final salvo em `features_eleitas.json`. Falta a F5 usá-lo.
- [x] Testes de voto, desempate e mínimo de features.
- [ ] PR #1 integrado à `main`.

## G5 · Dois algoritmos

Evidência: `evidencias/competicao/` (o modelo serializado fica em `models/candidatos/<cycle_id>/`, fora do git).

- [x] Dois algoritmos distintos no notebook 03 (A = regressão logística, B = random forest).
- [x] Mesmas features eleitas, mesmo treino, mesma partição e mesma métrica.
- [x] Folds gerados com `dividir_cv`.
- [x] Feature eleita ausente em um ajuste do one-hot é tratada: entra zerada e fica anotada em `features_ausentes`.
- [x] Limitação da média entre folds medida: CV com eleição refeita por fold, comparado ao CV oficial.
- [x] Modelo serializado junto com pré-processamento e features (um `Pipeline` por modelo).
- [ ] Dois runs comparáveis no MLflow, com o mesmo `cycle_id`. A célula já existe no notebook 03; falta a F6.
- [x] Hiperparâmetros, seeds, métricas e features em `resultados_competicao.json`. No MLflow, depende da F6.

## G6 · MLflow e rastreabilidade

Evidência: `evidencias/mlflow/` (2 ciclos locais com `run_id`s). Detalhes em [`f6_mlflow.md`](f6_mlflow.md). Falta a captura do MLflow no relatório.

- [x] Experimento, `run_id`, `cycle_id` e `data_version` em todo run. `iniciar_run` recusa contexto sem as duas tags (`tests/test_rastreio.py`).
- [x] Registro chamado no fim do notebook 02: rankings, votos, features eleitas, parâmetros e `cycle_id`.
- [ ] Algoritmos, parâmetros, métricas e modelos serializados registrados. **As funções existem e têm teste (`registrar_parametros`, `registrar_metricas`, `registrar_modelo`); falta a F5 chamá-las no notebook 03.**
- [ ] Decisão de promoção ou manutenção registrada. **`registrar_decisao` existe e tem teste; falta a F8 chamá-la no notebook 04.**
- [ ] Histórico sobrevive ao fim do runner, por artefato baixável ou armazenamento persistente. **O estado é uma pasta só (`mlflow_estado/`) e funciona em outro caminho (teste de relocação); falta a F7 guardá-la. Proposta: branch `estado-pipeline`.**
- [x] Pelo menos dois ciclos recuperáveis e consultáveis (`historico_ciclos`, `python -m src.rastreio`). Hoje só locais.
- [ ] José e Diego combinaram onde o estado fica guardado.

## G7 · Workflow único e agendamento

Evidência esperada: YAML, logs e link da execução no Actions. Como funciona: [`f7_workflow.md`](f7_workflow.md).

- [x] Só existe `.github/workflows/pipeline_mlops.yml`, com `schedule` (diário, 09:00 UTC) e `workflow_dispatch` (verificado em `tests/test_workflow.py`).
- [ ] Instala dependências e obtém dados e estado persistente. Implementado; falta rodar no Actions.
- [x] Executa 01→02→03→04→05 em sequência, um notebook por passo, sem `--allow-errors` (verificado em `tests/test_workflow.py`).
- [ ] Falha interrompe o fluxo e impede promoção indevida, com um teste real de falha. Implementado (entrada `falhar_em` do `workflow_dispatch`); falta a execução no Actions.
- [ ] Manifesto, modelo vigente e histórico são recuperados de armazenamento persistente (branch `estado-pipeline` e Release `modelos`). Lógica simulada localmente em 3 execuções; falta rodar no Actions.
- [ ] Publica o pacote completo de resultados. Implementado; falta rodar no Actions.
- [ ] **Execução real pelo gatilho `schedule`.** A manual não substitui. **Hoje há 0 execuções no Actions.**

> O `schedule` só dispara a partir da branch padrão, então o workflow precisa estar na `main`. O GitHub também pode atrasar os agendamentos: deixem folga antes da entrega. O repositório precisa de *Workflow permissions* = "Read and write".

## G8 · Promoção e previsões

Evidência esperada: `models/modelo_vigente.json`, previsões e decisões de 2 ciclos.

- [ ] Compara A e B automaticamente e elege o vencedor no notebook 04.
- [ ] Confronta o vencedor com o vigente em protocolo compatível, nos mesmos dados, e não com uma métrica antiga guardada.
- [ ] Limiar de melhoria e verificações de qualidade definidos antes, em `config.yaml`.
- [ ] Mantém o vigente quando os critérios não são atendidos.
- [ ] Manifesto traz identificação, versão dos dados, features, métrica, `run_id` e decisão.
- [ ] Modelos anteriores preservados e recuperáveis.
- [ ] Notebook 05 carrega o vigente com pré-processamento e features.
- [ ] Previsões em registros não usados na escolha do modelo.
- [ ] Testes de primeiro ciclo, melhora suficiente, melhora insuficiente e abaixo do piso.

## G9 · Repositório, relatório e evidências

Evidência esperada: `reports/relatorio_n3.pdf`, `README.md`, `HISTORICO_DESENVOLVIMENTO.md`.

- [ ] README explica instalação, execução local, execução agendada, consulta ao MLflow e reprodução das previsões. **Faltam agendamento, MLflow e previsões.**
- [ ] Seguir o README em um ambiente limpo funciona.
- [ ] Relatório em PDF identifica os integrantes.
- [ ] Relatório cobre problema e dados, ETL, tabela de votação e comparação dos dois modelos.
- [ ] Relatório cobre decisão de promoção, arquitetura do fluxo único e evidências de execução.
- [ ] Relatório cobre limitações e conclusões, incluindo as suposições sobre `adults`, `children`, `stays_in_*` e `meal`.
- [ ] Evidências do `schedule` e histórico do MLflow incluídos.
- [ ] `HISTORICO_DESENVOLVIMENTO.md` regenerado no fim.

## G10 · Recurso adicional

Evidência esperada: pasta em `evidencias/` e seção no relatório.

- [ ] Recurso escolhido e registrado, sem repetir os critérios 1 a 9. A validação de `validar_dados_brutos` faz parte da F2 e não conta.
- [ ] Responde a uma necessidade real do projeto, demonstrada com dados.
- [ ] Integrado ao fluxo ou ao uso do modelo, e não um script solto.
- [ ] Resultado demonstrado com execução real.
- [ ] Testes do recurso.
- [ ] Relatório traz problema identificado, justificativa, solução, ponto de integração e resultado.

## GF · Teste integrado final

É o último gate e repete a checagem final do enunciado.

- [ ] Ciclo inicial e ciclo com lote adicional executados, identificando o que mudou.
- [ ] Votação refeita em cada ciclo, com o mesmo `cycle_id` e `data_version` do início ao fim.
- [ ] Dois modelos treinados e decisão automática registrada em cada ciclo.
- [ ] Previsões com o modelo vigente, com logs sem erros.
- [ ] Execução por `schedule` comprovada.
- [ ] MLflow recupera os ciclos e suas decisões.
- [ ] Recurso adicional demonstrado.
- [ ] Teste só usado na avaliação final, e `python -m pytest -q` passando.
