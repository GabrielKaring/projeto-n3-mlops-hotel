# Gates de entrega do Trabalho N3

Um **gate** é o conjunto de condições verificáveis que uma feature precisa cumprir para ser considerada pronta. As condições vêm do enunciado e da apresentação do professor. Os detalhes de cada feature estão em [`divisao_de_tarefas.md`](divisao_de_tarefas.md), e os donos em [`config/equipe.yaml`](../config/equipe.yaml).

**Como usar**
- O dono marca uma condição só com a evidência em mãos: arquivo, log ou `run_id`. Código que existe, mas não roda, não conta.
- O gate só passa com todas as condições marcadas.
- Print complementa, mas não substitui código e logs verificáveis.

**Pontuação de cada critério (enunciado, seção 8):** 1,0 = implementado e demonstrado; 0,5 = parcial, com evidência insuficiente ou falha relevante; 0,0 = ausente, não executável ou não comprovado.

## Painel

Situação em 23/11/2026, na `main`.

| Gate | O que cobre | Dono | Depende de | Situação | O que falta |
|---|---|---|---|---|---|
| G0 | Regras do grupo e do repositório | Todos | | 🟡 parcial | Confirmar com o professor o grupo de 7 e dar acesso ao repositório |
| G1 | Problema, dados e avaliação | Gabriel Karing | | ✅ passou | |
| G2 | ETL e atributos | Gabriel Karing | G1 | ✅ passou | |
| G3 | Três métodos de seleção | Gabriel Karing | G2 | ✅ passou | |
| G4 | Votação | Gabriel Karing | G3 | ✅ passou | |
| G5 | Dois algoritmos | Gabriel Wendorff | G4 | ✅ passou | |
| G6 | MLflow, incluindo o registro de seleção e votação | José | G3, G4 | ✅ passou | |
| G7 | Workflow e agendamento | Diego | G6 | ✅ passou | |
| G8 | Promoção e previsões | Diego | G5, G6 | ✅ passou | |
| G9 | Repositório, relatório e evidências | Vinicius | todos | ✅ passou | Regenerar o `HISTORICO_DESENVOLVIMENTO.md` depois do último commit |
| G10 | Recurso adicional | Lucas e Guilherme | G2, G7 | ✅ passou | |
| GF | Teste integrado final | Todos | todos | ✅ passou | |

Ordem do trabalho: `G1 → G2 → G3 → G4 → G5 → G8 → G9 → GF`. G6 e G7 correm em paralelo desde o início, e G10 entra no fluxo pelo G7.

**Evidências usadas abaixo.** Execuções do Actions: [#2](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506), [#3](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45091260105) (falha simulada), [#4](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735), [#5](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641) e [#6](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) (`schedule`). A cópia do que elas produziram está em [`evidencias/actions/`](../evidencias/actions/), uma pasta por feature, e o manifesto em `models/modelo_vigente_actions.json`.

## G0 · Regras do grupo e do repositório

- [ ] Grupo com até 5 integrantes, ou exceção confirmada pelo professor. **Hoje são 7.**
- [ ] Repositório acessível ao professor. **Hoje é privado.**
- [ ] Todos os integrantes com acesso de escrita. **Hoje só há 2 colaboradores.**
- [x] Commits de cada pessoa com o nome de `config/equipe.yaml` ou com linha no `.mailmap`. O Lucas ainda não tem commit.
- [x] Sem tokens, senhas, ambientes virtuais ou dados restritos versionados.
- [x] Dados públicos, com fonte e licença indicadas.

## G1 · Problema, dados e avaliação

Evidência: `docs/etapa1_problema_dados_avaliacao.md`.

- [x] Tipo de problema escolhido e justificado.
- [x] Fonte, licença, unidade de observação, alvo, atributos e limitações descritos.
- [x] Métrica principal e direção definidas antes de treinar qualquer modelo.
- [x] Partição cronológica, com teste reservado à avaliação final.
- [x] Forma de chegada de novos registros explicada.

## G2 · ETL e engenharia de atributos

- [x] Ausentes, duplicatas, inconsistências e outliers verificados no notebook 01.
- [x] Tratamentos justificados e diagnóstico de antes e depois.
- [x] Transformações ajustadas só no treino, via `Pipeline`/`ColumnTransformer`.
- [x] Novos lotes identificados e incorporados sem duplicação, com `data_version`.
- [x] Ciclos refeitos no pipeline final, sem `country`: #2, #4, #5 e #6 (`dv019` a `dv022`).

## G3 · Três métodos de seleção

- [x] Três métodos no notebook 02, sobre os mesmos dados.
- [x] Ranking ou importância de cada método salvo.
- [x] Ajuste só no treino e dentro de cada fold, com teste de ausência de vazamento.
- [x] Cada método entrega a lista de features aprovadas.
- [x] PR #1 integrado à `main` (19/11).

## G4 · Votação e eleição

- [x] Regra definida antes, em `config.yaml`, e justificada no notebook 02.
- [x] Top-k, desempate e quantidade mínima de features aplicados; empatadas na fronteira recebem o voto.
- [x] Tabela de votos gerada a cada ciclo.
- [x] Um único conjunto final (`features_eleitas.json`), usado pelos dois modelos da F5.
- [x] Testes de voto, desempate e mínimo de features.

## G5 · Dois algoritmos

- [x] Dois algoritmos distintos no notebook 03 (A = regressão logística, B = random forest).
- [x] Mesmas features, treino, partição, folds (`dividir_cv`) e métrica.
- [x] Feature eleita ausente num fold entra zerada e fica anotada em `features_ausentes`.
- [x] Otimismo da média entre folds medido (CV com eleição refeita por fold).
- [x] Modelo serializado com pré-processamento e features (um `Pipeline` por modelo).
- [x] Dois runs comparáveis no MLflow por ciclo (`03_modelo_A` e `03_modelo_B`), com hiperparâmetros, seeds, métricas e features.

## G6 · MLflow e rastreabilidade

Detalhes em [`f6_mlflow.md`](f6_mlflow.md).

- [x] `run_id`, `cycle_id` e `data_version` em todo run; `iniciar_run` recusa contexto sem as tags.
- [x] Seleção e votação registradas no notebook 02; modelos no 03; decisão no 04 (tag `decisao`); drift no run `01_drift`.
- [x] Histórico sobrevive ao fim do runner: `mlflow_estado/` vai para a branch `estado-pipeline` a cada execução.
- [x] Ciclos recuperáveis e consultáveis (`python -m src.rastreio`): 4 completos e 1 com falha no Actions.

## G7 · Workflow único e agendamento

Como funciona: [`f7_workflow.md`](f7_workflow.md).

- [x] Só existe `.github/workflows/pipeline_mlops.yml`, com `schedule` e `workflow_dispatch` (`tests/test_workflow.py`).
- [x] Instala dependências e recupera dados, estado e vigente (branch `estado-pipeline` e Release `modelos`).
- [x] Executa 01→02→03→04→05, um notebook por passo, sem `--allow-errors`.
- [x] Falha interrompe o fluxo e impede promoção: #3 parou no 03, e o 04 e a Release não rodaram.
- [x] Publica as evidências como artefato `evidencias-<run_id>`.
- [x] **Execução real pelo `schedule`:** #6, com todos os passos verdes.

> O `schedule` só dispara a partir da branch padrão, pode atrasar e exige *Workflow permissions* = "Read and write".

## G8 · Promoção e previsões

- [x] Vencedor eleito no notebook 04 pela maior média de `f1_macro` no CV; empate vai para o menor desvio.
- [x] Vencedor e vigente avaliados no mesmo teste do ciclo atual.
- [x] Melhoria mínima (0,005) e piso (0,60) definidos antes, em `config.yaml`.
- [x] Mantém o vigente quando a regra não é atendida: #5 e #6.
- [x] Manifesto com identificação, versão dos dados, features, métrica, `run_id`, decisão e histórico.
- [x] Modelos anteriores preservados na Release `modelos`, sem sobrescrita.
- [x] Notebook 05 carrega o vigente completo e prevê o próximo lote de `data/incoming/`, conferido quando ele chega.
- [x] Testes de primeiro ciclo, melhora suficiente, insuficiente e abaixo do piso (`tests/test_promocao.py`).

## G9 · Repositório, relatório e evidências

- [x] README com instalação, execução local e agendada, MLflow, previsões e demonstração com uma reserva nova.
- [x] Seguir o README em ambiente limpo funciona (GF).
- [x] Relatório em PDF com integrantes, dados, ETL, votação, modelos, promoção, fluxo, evidências, limitações e conclusões.
- [x] Evidências do `schedule` e histórico do MLflow na `main` (`evidencias/actions/`).
- [ ] `HISTORICO_DESENVOLVIMENTO.md` regenerado depois do último commit (`python scripts/gerar_historico.py`).

## G10 · Recurso adicional

Monitoramento de drift com alerta: [`f10_drift.md`](f10_drift.md) e seção 9 do relatório.

- [x] Não repete os critérios 1 a 9.
- [x] Responde a uma necessidade real, mostrada com dados (cancelamento de 2017 acima do de 2016).
- [x] Integrado ao fluxo: passo logo depois do notebook 01, com `::warning`, resumo do job e run no MLflow.
- [x] Demonstrado nas execuções do Actions e com testes (`tests/test_drift.py`).

## GF · Teste integrado final

Teste de 23/11/2026 numa cópia limpa da `main` (`git archive` de `73d525c`), seguindo o README: 2 ciclos locais que reproduziram bit a bit o F1 de #2 e #4, mais um ciclo com `data/incoming` vazio, sem erro.

- [x] Ciclo inicial e ciclo com lote adicional, identificando o que mudou.
- [x] Votação refeita em cada ciclo, com o mesmo `cycle_id` e `data_version` do início ao fim.
- [x] Dois modelos treinados e decisão automática registrada em cada ciclo.
- [x] Previsões com o vigente, sem erro nos logs.
- [x] Execução por `schedule` comprovada (#6).
- [x] MLflow recupera os ciclos e as decisões.
- [x] Recurso adicional demonstrado.
- [x] Teste só na avaliação final, e `python -m pytest -q` passando.
