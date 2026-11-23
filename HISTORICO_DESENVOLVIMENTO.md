# Histórico de desenvolvimento

_Gerado em 23/11/2026 por `scripts/gerar_historico.py` a partir do `git log`. Não edite à mão: rode o script de novo._

## Resumo por integrante

| Integrante | Commits | Primeiro | Último | Linhas + | Linhas − |
|---|---:|---|---|---:|---:|
| Gabriel Karing | 22 | 2026-11-04 | 2026-11-23 | 15,828 | 11,376 |
| Vinicius | 15 | 2026-11-23 | 2026-11-23 | 1,440 | 198 |
| Gabriel Wendorff | 3 | 2026-11-19 | 2026-11-19 | 1,497 | 39 |
| Lucas | 3 | 2026-11-23 | 2026-11-23 | 2,311 | 4 |
| Diego | 2 | 2026-11-19 | 2026-11-19 | 11,338 | 99 |
| Guilherme | 2 | 2026-11-21 | 2026-11-21 | 840 | 5 |
| José | 1 | 2026-11-19 | 2026-11-19 | 6,022 | 35 |

Total: 48 commits de 7 integrante(s).

## Visão por feature

Commits com a tag `[Fn]` no início da mensagem. "Do dono" conta commits cujo autor é o dono da feature em `config/equipe.yaml`; "de apoio" são os de outras pessoas na mesma feature.

| Feature | Título | Dono | Commits do dono | Commits de apoio | Situação |
|---|---|---|---:|---:|---|
| F1 | Problema, dados e avaliação | Gabriel Karing | 1 | 0 | ✅ com commits do dono |
| F2 | ETL e engenharia de atributos | Gabriel Karing | 4 | 0 | ✅ com commits do dono |
| F3 | Três métodos de seleção de features | Gabriel Karing | 3 | 0 | ✅ com commits do dono |
| F4 | Votação e eleição das features | Gabriel Karing | 2 | 0 | ✅ com commits do dono |
| F5 | Treinamento dos dois algoritmos | Gabriel Wendorff | 3 | 1 | ✅ com commits do dono |
| F6 | MLflow e rastreabilidade | José | 1 | 2 | ✅ com commits do dono |
| F7 | Workflow único e agendamento | Diego | 0 | 4 | ⬜ sem commits do dono |
| F8 | Eleição, promoção e previsões | Diego | 0 | 1 | ⬜ sem commits do dono |
| F9 | Repositório, relatório e evidências | Vinicius | 7 | 3 | ✅ com commits do dono |
| F10 | Recurso adicional justificado (sugestão: monitoramento de drift com alerta) | Lucas, Guilherme | 5 | 2 | ✅ com commits do dono |

### Por integrante

| Integrante | Features | Commits nas próprias features |
|---|---|---:|
| Gabriel Karing | F1, F2, F3, F4 | 9 |
| Diego | F7, F8 | 0 |
| Gabriel Wendorff | F5 | 3 |
| Guilherme | F10 | 2 |
| José | F6 | 1 |
| Lucas | F10 | 3 |
| Vinicius | F9 | 7 |

13 commit(s) sem tag de feature (docs, equipe, correções).

#### F1 · Problema, dados e avaliação

- `7ca82cf` 2026-11-04 · Gabriel Karing · [F1] define problema, dados, métrica e protocolo de avaliação

#### F2 · ETL e engenharia de atributos

- `709df2e` 2026-11-04 · Gabriel Karing · [F2] ingestão em lotes mensais com manifesto, sha256 e data_version
- `9fd2630` 2026-11-04 · Gabriel Karing · [F2] ETL, engenharia de atributos, partição temporal e protocolo de avaliação
- `99816f1` 2026-11-04 · Gabriel Karing · [F2] notebook 01 executado em dois ciclos (dv018 e dv019)
- `e443c57` 2026-11-23 · Gabriel Karing · [F2][F8] demonstração de uso com reserva nova e análise de erros

#### F3 · Três métodos de seleção de features

- `52b66c2` 2026-11-05 · Gabriel Karing · [F3] implementa os três métodos de seleção de features
- `5bf540a` 2026-11-05 · Gabriel Karing · [F3] aplica a revisão: retira country, valida a ordem e redefine os donos
- `70d55b0` 2026-11-05 · Gabriel Karing · [F3][F4] adiciona as listas de aprovadas e implementa a votação

#### F4 · Votação e eleição das features

- `70d55b0` 2026-11-05 · Gabriel Karing · [F3][F4] adiciona as listas de aprovadas e implementa a votação
- `69f4175` 2026-11-05 · Gabriel Karing · [F4][F6] passa para a F6 o registro da seleção e da votação no MLflow

#### F5 · Treinamento dos dois algoritmos

- `f09ce5f` 2026-11-19 · Gabriel Wendorff · [F5] implementa o treino e a avaliação dos modelos A e B
- `119a540` 2026-11-19 · Gabriel Wendorff · [F5] notebook 03 com a competição e a medição do otimismo do CV
- `b6bff05` 2026-11-19 · Gabriel Wendorff · [F5] evidências do primeiro ciclo e atualização do G5
- `1d5f932` 2026-11-23 · Vinicius · [F5][F6] grava os candidatos comprimidos para o histórico do MLflow não crescer 50 MB por ciclo

#### F6 · MLflow e rastreabilidade

- `69f4175` 2026-11-05 · Gabriel Karing · [F4][F6] passa para a F6 o registro da seleção e da votação no MLflow
- `62ca370` 2026-11-19 · José · [F6] implementa o registro no MLflow e a consulta dos ciclos
- `1d5f932` 2026-11-23 · Vinicius · [F5][F6] grava os candidatos comprimidos para o histórico do MLflow não crescer 50 MB por ciclo

#### F7 · Workflow único e agendamento

- `bbf243f` 2026-11-23 · Vinicius · [F7] agendamento temporário de hora em hora para gerar os ciclos antes da entrega
- `f983062` 2026-11-23 · Vinicius · [F7] corrige o tamanho do estado no doc depois da compressão
- `67523ef` 2026-11-23 · Vinicius · [F7] agendamento temporário a cada 10 min
- `90272e1` 2026-11-23 · Vinicius · [F7] tira o agendamento temporário, fica só o diário

#### F8 · Eleição, promoção e previsões

- `e443c57` 2026-11-23 · Gabriel Karing · [F2][F8] demonstração de uso com reserva nova e análise de erros

#### F9 · Repositório, relatório e evidências

- `f57facc` 2026-11-23 · Vinicius · [F9] reconhece os commits do Diego no histórico (.mailmap)
- `5998018` 2026-11-23 · Vinicius · [F9] atualiza o README com agendamento, MLflow e previsões
- `528715b` 2026-11-23 · Vinicius · [F9] script que gera o relatório em PDF a partir do markdown
- `85e874c` 2026-11-23 · Vinicius · [F9] relatório final com os ciclos do Actions
- `7b00b27` 2026-11-23 · Vinicius · [F9] regenera o histórico de desenvolvimento
- `0178de1` 2026-11-23 · Vinicius · [F9] relatório com o ciclo agendado
- `73d525c` 2026-11-23 · Vinicius · [F9] regenera o histórico de desenvolvimento
- `acd6f6f` 2026-11-23 · Gabriel Karing · [F9] relatório, README e evidências do Actions na main
- `201b7c5` 2026-11-23 · Gabriel Karing · [F9] regenera o histórico de desenvolvimento
- `75b71fb` 2026-11-23 · Gabriel Karing · [F9] introdução, título com as duas disciplinas e G0 resolvido

#### F10 · Recurso adicional justificado (sugestão: monitoramento de drift com alerta)

- `9ccf80e` 2026-11-21 · Guilherme · [F10] fixa a regra do monitoramento de drift antes de rodar nos dados
- `c48af1d` 2026-11-21 · Guilherme · [F10] implementa o monitoramento de drift com alerta, testes e documentação
- `c01be7e` 2026-11-23 · Vinicius · [F10] liga o monitoramento de drift no workflow, logo depois do notebook 01
- `89738e9` 2026-11-23 · Gabriel Karing · [F10] drift sem alarme por troca de agências e registrado no MLflow
- `9412c30` 2026-11-23 · Lucas · [F10] teste que confere o passo do drift no workflow
- `3848cc7` 2026-11-23 · Lucas · [F10] evidências do drift nos oito lotes de 2017 com a regra atual
- `2f7af11` 2026-11-23 · Lucas · [F10] aponta o teste do workflow e as evidências locais no f10_drift.md

## Linha do tempo

| Data | Commit | Integrante | Mensagem | Arquivos |
|---|---|---|---|---:|
| 2026-11-04 | `7ca82cf` | Gabriel Karing | [F1] define problema, dados, métrica e protocolo de avaliação | 8 |
| 2026-11-04 | `709df2e` | Gabriel Karing | [F2] ingestão em lotes mensais com manifesto, sha256 e data_version | 36 |
| 2026-11-04 | `9fd2630` | Gabriel Karing | [F2] ETL, engenharia de atributos, partição temporal e protocolo de avaliação | 4 |
| 2026-11-04 | `99816f1` | Gabriel Karing | [F2] notebook 01 executado em dois ciclos (dv018 e dv019) | 8 |
| 2026-11-04 | `97b9fe7` | Gabriel Karing | [equipe] esqueletos das etapas 3 a 8, workflow mínimo e divisão de tarefas | 9 |
| 2026-11-04 | `cc34665` | Gabriel Karing | [docs] README e gerador do histórico de desenvolvimento | 2 |
| 2026-11-04 | `13301a7` | Gabriel Karing | [fix] limita pandas<3 (mlflow) e troca dt.dayofweek por day_of_week | 2 |
| 2026-11-04 | `0ac63d7` | Gabriel Karing | [docs] atualiza histórico de desenvolvimento | 1 |
| 2026-11-04 | `00f5c05` | Gabriel Karing | [equipe] divide o projeto em features F1-F10 com dono (1 a 2 por pessoa) | 18 |
| 2026-11-04 | `41616e7` | Gabriel Karing | [docs] regenera histórico com visão por feature | 1 |
| 2026-11-05 | `52b66c2` | Gabriel Karing | [F3] implementa os três métodos de seleção de features | 10 |
| 2026-11-05 | `5bf540a` | Gabriel Karing | [F3] aplica a revisão: retira country, valida a ordem e redefine os donos | 21 |
| 2026-11-05 | `c445d08` | Gabriel Karing | [equipe] usa o primeiro nome dos integrantes e distingue os dois Gabriel | 4 |
| 2026-11-05 | `dcbfb31` | Gabriel Karing | [equipe] concentra as features críticas finais em Diego e Vinicius | 3 |
| 2026-11-05 | `f9b4929` | Gabriel Karing | [docs] adiciona os gates de entrega a partir do enunciado | 2 |
| 2026-11-05 | `70d55b0` | Gabriel Karing | [F3][F4] adiciona as listas de aprovadas e implementa a votação | 17 |
| 2026-11-05 | `69f4175` | Gabriel Karing | [F4][F6] passa para a F6 o registro da seleção e da votação no MLflow | 9 |
| 2026-11-19 | `f09ce5f` | Gabriel Wendorff | [F5] implementa o treino e a avaliação dos modelos A e B | 2 |
| 2026-11-19 | `119a540` | Gabriel Wendorff | [F5] notebook 03 com a competição e a medição do otimismo do CV | 2 |
| 2026-11-19 | `b6bff05` | Gabriel Wendorff | [F5] evidências do primeiro ciclo e atualização do G5 | 5 |
| 2026-11-19 | `62ca370` | José | [F6] implementa o registro no MLflow e a consulta dos ciclos | 13 |
| 2026-11-19 | `ab04e71` | Diego | Implementa promoção e previsões do modelo vigente | 17 |
| 2026-11-19 | `db0c217` | Diego | Implementar workflow MLOps completo | 6 |
| 2026-11-21 | `9ccf80e` | Guilherme | [F10] fixa a regra do monitoramento de drift antes de rodar nos dados | 1 |
| 2026-11-21 | `c48af1d` | Guilherme | [F10] implementa o monitoramento de drift com alerta, testes e documentação | 7 |
| 2026-11-23 | `c01be7e` | Vinicius | [F10] liga o monitoramento de drift no workflow, logo depois do notebook 01 | 2 |
| 2026-11-23 | `f57facc` | Vinicius | [F9] reconhece os commits do Diego no histórico (.mailmap) | 1 |
| 2026-11-23 | `1d5f932` | Vinicius | [F5][F6] grava os candidatos comprimidos para o histórico do MLflow não crescer 50 MB por ciclo | 1 |
| 2026-11-23 | `bbf243f` | Vinicius | [F7] agendamento temporário de hora em hora para gerar os ciclos antes da entrega | 1 |
| 2026-11-23 | `5998018` | Vinicius | [F9] atualiza o README com agendamento, MLflow e previsões | 1 |
| 2026-11-23 | `f983062` | Vinicius | [F7] corrige o tamanho do estado no doc depois da compressão | 1 |
| 2026-11-23 | `528715b` | Vinicius | [F9] script que gera o relatório em PDF a partir do markdown | 1 |
| 2026-11-23 | `67523ef` | Vinicius | [F7] agendamento temporário a cada 10 min | 1 |
| 2026-11-23 | `85e874c` | Vinicius | [F9] relatório final com os ciclos do Actions | 5 |
| 2026-11-23 | `e7a3b62` | Vinicius | [docs] atualiza o status das features e dos gates | 2 |
| 2026-11-23 | `7b00b27` | Vinicius | [F9] regenera o histórico de desenvolvimento | 1 |
| 2026-11-23 | `90272e1` | Vinicius | [F7] tira o agendamento temporário, fica só o diário | 1 |
| 2026-11-23 | `0178de1` | Vinicius | [F9] relatório com o ciclo agendado | 2 |
| 2026-11-23 | `0ff2b4c` | Vinicius | [docs] marca o G7 como passou | 2 |
| 2026-11-23 | `73d525c` | Vinicius | [F9] regenera o histórico de desenvolvimento | 1 |
| 2026-11-23 | `e443c57` | Gabriel Karing | [F2][F8] demonstração de uso com reserva nova e análise de erros | 13 |
| 2026-11-23 | `89738e9` | Gabriel Karing | [F10] drift sem alarme por troca de agências e registrado no MLflow | 7 |
| 2026-11-23 | `acd6f6f` | Gabriel Karing | [F9] relatório, README e evidências do Actions na main | 39 |
| 2026-11-23 | `201b7c5` | Gabriel Karing | [F9] regenera o histórico de desenvolvimento | 1 |
| 2026-11-23 | `75b71fb` | Gabriel Karing | [F9] introdução, título com as duas disciplinas e G0 resolvido | 5 |
| 2026-11-23 | `9412c30` | Lucas | [F10] teste que confere o passo do drift no workflow | 1 |
| 2026-11-23 | `3848cc7` | Lucas | [F10] evidências do drift nos oito lotes de 2017 com a regra atual | 27 |
| 2026-11-23 | `2f7af11` | Lucas | [F10] aponta o teste do workflow e as evidências locais no f10_drift.md | 1 |

## Contribuições de Gabriel Karing

- `7ca82cf` 2026-11-04 · [F1] define problema, dados, métrica e protocolo de avaliação
- `709df2e` 2026-11-04 · [F2] ingestão em lotes mensais com manifesto, sha256 e data_version
- `9fd2630` 2026-11-04 · [F2] ETL, engenharia de atributos, partição temporal e protocolo de avaliação
- `99816f1` 2026-11-04 · [F2] notebook 01 executado em dois ciclos (dv018 e dv019)
- `97b9fe7` 2026-11-04 · [equipe] esqueletos das etapas 3 a 8, workflow mínimo e divisão de tarefas
- `cc34665` 2026-11-04 · [docs] README e gerador do histórico de desenvolvimento
- `13301a7` 2026-11-04 · [fix] limita pandas<3 (mlflow) e troca dt.dayofweek por day_of_week
- `0ac63d7` 2026-11-04 · [docs] atualiza histórico de desenvolvimento
- `00f5c05` 2026-11-04 · [equipe] divide o projeto em features F1-F10 com dono (1 a 2 por pessoa)
- `41616e7` 2026-11-04 · [docs] regenera histórico com visão por feature
- `52b66c2` 2026-11-05 · [F3] implementa os três métodos de seleção de features
- `5bf540a` 2026-11-05 · [F3] aplica a revisão: retira country, valida a ordem e redefine os donos
- `c445d08` 2026-11-05 · [equipe] usa o primeiro nome dos integrantes e distingue os dois Gabriel
- `dcbfb31` 2026-11-05 · [equipe] concentra as features críticas finais em Diego e Vinicius
- `f9b4929` 2026-11-05 · [docs] adiciona os gates de entrega a partir do enunciado
- `70d55b0` 2026-11-05 · [F3][F4] adiciona as listas de aprovadas e implementa a votação
- `69f4175` 2026-11-05 · [F4][F6] passa para a F6 o registro da seleção e da votação no MLflow
- `e443c57` 2026-11-23 · [F2][F8] demonstração de uso com reserva nova e análise de erros
- `89738e9` 2026-11-23 · [F10] drift sem alarme por troca de agências e registrado no MLflow
- `acd6f6f` 2026-11-23 · [F9] relatório, README e evidências do Actions na main
- `201b7c5` 2026-11-23 · [F9] regenera o histórico de desenvolvimento
- `75b71fb` 2026-11-23 · [F9] introdução, título com as duas disciplinas e G0 resolvido

## Contribuições de Gabriel Wendorff

- `f09ce5f` 2026-11-19 · [F5] implementa o treino e a avaliação dos modelos A e B
- `119a540` 2026-11-19 · [F5] notebook 03 com a competição e a medição do otimismo do CV
- `b6bff05` 2026-11-19 · [F5] evidências do primeiro ciclo e atualização do G5

## Contribuições de José

- `62ca370` 2026-11-19 · [F6] implementa o registro no MLflow e a consulta dos ciclos

## Contribuições de Diego

- `ab04e71` 2026-11-19 · Implementa promoção e previsões do modelo vigente
- `db0c217` 2026-11-19 · Implementar workflow MLOps completo

## Contribuições de Guilherme

- `9ccf80e` 2026-11-21 · [F10] fixa a regra do monitoramento de drift antes de rodar nos dados
- `c48af1d` 2026-11-21 · [F10] implementa o monitoramento de drift com alerta, testes e documentação

## Contribuições de Vinicius

- `c01be7e` 2026-11-23 · [F10] liga o monitoramento de drift no workflow, logo depois do notebook 01
- `f57facc` 2026-11-23 · [F9] reconhece os commits do Diego no histórico (.mailmap)
- `1d5f932` 2026-11-23 · [F5][F6] grava os candidatos comprimidos para o histórico do MLflow não crescer 50 MB por ciclo
- `bbf243f` 2026-11-23 · [F7] agendamento temporário de hora em hora para gerar os ciclos antes da entrega
- `5998018` 2026-11-23 · [F9] atualiza o README com agendamento, MLflow e previsões
- `f983062` 2026-11-23 · [F7] corrige o tamanho do estado no doc depois da compressão
- `528715b` 2026-11-23 · [F9] script que gera o relatório em PDF a partir do markdown
- `67523ef` 2026-11-23 · [F7] agendamento temporário a cada 10 min
- `85e874c` 2026-11-23 · [F9] relatório final com os ciclos do Actions
- `e7a3b62` 2026-11-23 · [docs] atualiza o status das features e dos gates
- `7b00b27` 2026-11-23 · [F9] regenera o histórico de desenvolvimento
- `90272e1` 2026-11-23 · [F7] tira o agendamento temporário, fica só o diário
- `0178de1` 2026-11-23 · [F9] relatório com o ciclo agendado
- `0ff2b4c` 2026-11-23 · [docs] marca o G7 como passou
- `73d525c` 2026-11-23 · [F9] regenera o histórico de desenvolvimento

## Contribuições de Lucas

- `9412c30` 2026-11-23 · [F10] teste que confere o passo do drift no workflow
- `3848cc7` 2026-11-23 · [F10] evidências do drift nos oito lotes de 2017 com a regra atual
- `2f7af11` 2026-11-23 · [F10] aponta o teste do workflow e as evidências locais no f10_drift.md
