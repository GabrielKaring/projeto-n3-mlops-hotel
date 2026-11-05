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
| G3 | Três métodos de seleção | Gabriel Karing | G2 | 🟡 parcial | Registro no MLflow, merge |
| G4 | Votação | Gabriel Karing | G3 | 🟡 parcial | Registro no MLflow, merge |
| G5 | Dois algoritmos | Gabriel Wendorff | G4 | ⬜ não iniciado | Tudo |
| G6 | MLflow | José | | ⬜ não iniciado | Tudo |
| G7 | Workflow e agendamento | Diego | G6 | 🟡 esqueleto | Agendamento, encadeamento, persistência |
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
- [ ] Rankings e parâmetros registrados no MLflow com `cycle_id` (depende do G6).
- [ ] PR #1 integrado à `main`.

## G4 · Votação e eleição

Evidência: `evidencias/votacao/`.

- [x] Regra definida antes da comparação, em `config.yaml`, e justificada no notebook 02.
- [x] Top-k, regra de desempate e quantidade mínima de features aplicados.
- [x] O empate na fronteira do top-k está tratado: as empatadas recebem o voto.
- [x] Tabela de votos por feature gerada a cada ciclo.
- [x] Um único conjunto final salvo em `features_eleitas.json`. Falta a F5 usá-lo.
- [ ] Votos, conjunto final, parâmetros e `cycle_id` registrados no MLflow.
- [x] Testes de voto, desempate e mínimo de features.
- [ ] PR #1 integrado à `main`.

## G5 · Dois algoritmos

Evidência esperada: `models/candidatos/<cycle_id>/`, `resultados_competicao.json`.

- [ ] Dois algoritmos distintos no notebook 03.
- [ ] Mesmas features eleitas, mesmo treino, mesma partição e mesma métrica.
- [ ] Folds gerados com `dividir_cv`.
- [ ] Feature eleita ausente em um ajuste do one-hot é tratada. Neste ciclo todas as eleitas existem nos 4 folds, mas isso pode mudar.
- [ ] Limitação da média entre folds medida ou declarada: o conjunto eleito já viu parte das validações.
- [ ] Modelo serializado junto com pré-processamento e features.
- [ ] Dois runs comparáveis no MLflow, com o mesmo `cycle_id`.
- [ ] Hiperparâmetros, seeds, métricas, features e artefatos registrados.

## G6 · MLflow e rastreabilidade

Evidência esperada: captura do MLflow e `run_id`s no relatório.

- [ ] Experimento, `run_id`, `cycle_id` e `data_version` em todo run.
- [ ] Rankings, votação e features eleitas registrados.
- [ ] Algoritmos, parâmetros, métricas e modelos serializados registrados.
- [ ] Decisão de promoção ou manutenção registrada.
- [ ] Histórico sobrevive ao fim do runner, por artefato baixável ou armazenamento persistente.
- [ ] Pelo menos dois ciclos recuperáveis e consultáveis.
- [ ] José e Diego combinaram onde o estado fica guardado.

## G7 · Workflow único e agendamento

Evidência esperada: YAML, logs e link da execução no Actions.

- [ ] Só existe `.github/workflows/pipeline_mlops.yml`, com `schedule` e `workflow_dispatch`. **Hoje o `schedule` está comentado.**
- [ ] Instala dependências e obtém dados e estado persistente.
- [ ] Executa 01→02→03→04 em sequência, sem `--allow-errors`; o 05 pode rodar ao final.
- [ ] Falha interrompe o fluxo e impede promoção indevida, com um teste real de falha.
- [ ] Manifesto, modelo vigente e histórico são recuperados de armazenamento persistente.
- [ ] Publica o pacote completo de resultados.
- [ ] **Execução real pelo gatilho `schedule`.** A manual não substitui. **Hoje há 0 execuções no Actions.**

> O `schedule` só dispara a partir da branch padrão, então o workflow precisa estar na `main`. O GitHub também pode atrasar os agendamentos: deixem folga antes da entrega.

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
