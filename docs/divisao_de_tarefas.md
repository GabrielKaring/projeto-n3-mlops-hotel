# Divisão de tarefas por feature

Cada **feature** corresponde a um critério do enunciado (F1 = critério 1, ..., F10 = critério 10) e tem **um dono** (a F10 é dividida por dois), que responde por ela na avaliação individual. Cada integrante tem de **1 a 4 features**.

A fonte única de "quem é dono de quê" é [`config/equipe.yaml`](../config/equipe.yaml). O teste `tests/test_equipe.py` falha se alguma feature ficar sem dono ou se alguém passar de 4 (ou ficar com 0).

> **Atenção:** o enunciado (capa) diz *"grupos de até 5 integrantes"*. Com 7 pessoas, confirmem com o professor antes da entrega. Com 5 pessoas, basta remover dois integrantes de `equipe.yaml` e redistribuir.

## Quem entrega o quê

Os donos de F5 a F10 são uma proposta até serem confirmados. A parte crítica final (F7, F8 e F9) fica com Diego e Vinicius.

| Feature | Critério | Dono (proposta) | Entrega | Status |
|---|---|---|---|---|
| **F1** | 1 · Problema, dados e avaliação | Gabriel Karing | `docs/etapa1_problema_dados_avaliacao.md`, `src/avaliacao.py`, `config.yaml` (dados, avaliação) | ✅ pronto |
| **F2** | 2 · ETL e engenharia de atributos | Gabriel Karing | `src/ingestao.py`, `src/preprocessamento.py`, `notebooks/01_etl.ipynb` | ✅ pronto |
| **F3** | 3 · Três métodos de seleção | Gabriel Karing | `src/features.py`, seção F3 do `notebooks/02_votacao_features.ipynb` | ✅ pronto |
| **F4** | 4 · Votação e eleição | Gabriel Karing | `src/votacao.py`, seção F4 do `notebooks/02_votacao_features.ipynb` | ✅ pronto |
| **F5** | 5 · Dois algoritmos disputam | Gabriel Wendorff | `src/modelagem.py`, `notebooks/03_treino_competicao.ipynb` | ✅ pronto (falta o registro no MLflow, que depende da F6) |
| **F6** | 6 · MLflow e rastreabilidade | José | `src/rastreio.py`, `config.yaml` (mlflow), seção F6 do `notebooks/02_votacao_features.ipynb`, `docs/f6_mlflow.md` | 🟨 falta a persistência no Actions (com F7) |
| **F7** | 7 · Workflow único e agendamento | Diego | `.github/workflows/pipeline_mlops.yml` | 🟨 esqueleto |
| **F8** | 8 · Eleição, promoção e previsões | Diego | `src/promocao.py`, `notebooks/04_*`, `notebooks/05_*` | 🟨 pronto localmente (falta guardar os modelos na Release, com F7) |
| **F9** | 9 · Repositório, relatório e evidências | Vinicius | `README.md`, `reports/relatorio_n3.pdf`, `evidencias/` | 🟨 README e histórico prontos; falta o relatório |
| **F10** | 10 · Recurso adicional | Lucas e Guilherme | `src/drift.py`, `tests/test_drift.py`, `docs/f10_drift.md`, `evidencias/drift/` | 🟨 em andamento |

Para trocar um dono: edite o campo `dono` em `config/equipe.yaml`, rode `python -m pytest tests/test_equipe.py` e faça um commit `[equipe] ...`. O nome em `integrantes` deve ser **igual ao nome dos seus commits no git** (ver `.mailmap`); é assim que o histórico por feature reconhece o seu trabalho.

## Definição de pronto e evidência de cada feature

O dono mostra a evidência no relatório e na apresentação. "Teste" é o que o dono adiciona em `tests/`. O checklist completo de cada feature, com a situação atual, está em [`gates.md`](gates.md).

| Feature | Pronto quando | Evidência | Teste mínimo |
|---|---|---|---|
| F1 | Problema, variável-alvo, métrica, partição e protocolo definidos antes de rodar modelos | `docs/etapa1_...md` | `tests/test_avaliacao.py` (existente) |
| F2 | Lotes versionados, validação, limpeza sem vazamento, `cycle_id` e `data_version` | `evidencias/etl/` (2 ciclos) | `tests/test_ingestao.py`, `test_preprocessamento.py` (existentes) |
| F3 | SelectKBest, importância do RF e RFE rodam sobre o mesmo X/y, ajustados só no treino/folds, rankings de cada método salvos | Tabela de scores por método no notebook 02 | `tests/test_features.py`: três rankings com o mesmo índice; sem uso do teste |
| F4 | Regra de votação definida **antes** em `config.yaml`; tabela de votos, desempate e mínimo de features; `features_eleitas.json` gerado | `evidencias/votacao/votos_<cycle_id>.csv` | `tests/test_votacao.py`: voto, desempate e piso de features em casos pequenos |
| F5 | Modelos A e B sobre as mesmas features, partição e métrica; cada um é um `Pipeline` completo serializado | `models/candidatos/<cycle_id>/`, `resultados_competicao.json` | `tests/test_modelagem.py`: A e B treinam e prevêem em dados sintéticos; mesmo protocolo |
| F6 | Todo run com `cycle_id`, `data_version`, parâmetros, métricas, features e artefatos; rankings e votação do notebook 02 registrados; dois ciclos recuperáveis; armazenamento persiste entre execuções do runner | Captura do MLflow com 2 ciclos + `run_id`s no relatório | `tests/test_rastreio.py`: run local com as tags do ciclo |
| F7 | Um único YAML com `schedule`, notebooks 01→05 encadeados, falha interrompe, estado persistido entre execuções | Log de execução **agendada** (não só `workflow_dispatch`) | Execuções reais no Actions (anexar link/log) |
| F8 | Comparação automática A×B, regra de promoção/manutenção, vigente preservado, modelos antigos mantidos, previsões com o vigente e seu pré-processamento | `models/modelo_vigente.json`, previsões, decisão em 2 ciclos | `tests/test_promocao.py`: primeiro ciclo, melhora suficiente, melhora insuficiente, abaixo do piso |
| F9 | README reproduzível, relatório PDF com a tabela de votação, a decisão de promoção e as limitações, histórico de commits por integrante | `reports/relatorio_n3.pdf`, `HISTORICO_DESENVOLVIMENTO.md` | Seguir o README em ambiente limpo |
| F10 | Necessidade real, integrado ao fluxo, resultado demonstrado e **não repete** os critérios 1 a 9 | `evidencias/drift/` + seção do relatório | `tests/test_drift.py` |

**Sugestão para F10:** monitoramento de drift com alerta. O dataset tem mudança real de distribuição (a taxa de cancelamento sobe para ~43% em abr-jun/2017), então há um problema concreto para motivar o recurso. A validação de esquema que já existe em `validar_dados_brutos` faz parte da F2 e não conta.

## Dependências entre as features

```
F1,F2,F3,F4 (prontos) ───────────> F5 ──> F8
                                           ▲
F6 (MLflow) registra F4, F5 e F8 ──────────┘
F7 (workflow) executa 01 → 05 e persiste o estado entre execuções
F10 (drift) lê os dados de F2 e alerta no fluxo de F7
```

Pares que precisam conversar cedo (as interfaces são os contratos nas docstrings de `src/`):
- **F3 → F4:** F3 devolve `{método: Series}` com o mesmo índice e `listas_aprovadas`; F4 consome.
- **F4 → F5:** F5 lê `data/processed/features_eleitas.json`.
- **F6 ↔ F7:** o runner do GitHub Actions é efêmero, então o MLflow, o manifesto dos lotes e o `modelo_vigente.json` precisam de armazenamento persistente. José (F6) e Diego (F7) precisam combinar isso cedo, porque F6 e F7 agora têm donos diferentes.
- **F5 → F8:** F8 lê os modelos e métricas em `models/candidatos/` e `resultados_competicao.json`.

O notebook 02 é compartilhado por F3 e F4.

## O que já está pronto e como usar

```python
from src.config import carregar_config, carregar_contexto
from src.preprocessamento import carregar_particao, construir_preprocessador, lista_features
from src.avaliacao import dividir_cv, metricas_classificacao, melhorou

cfg = carregar_config()
ctx = carregar_contexto(cfg)            # cycle_id, data_version, colunas, caminhos
dev = carregar_particao("dev", cfg)     # ordenado por data de chegada
pre = construir_preprocessador(cfg)     # NÃO ajustado: ajuste só no treino / dentro de cada fold
```

Regras que valem para todos (vêm do enunciado):
- A métrica (`f1_macro`) e o protocolo (`dividir_cv`) estão em `config/config.yaml`. Não troquem depois de ver resultados.
- O conjunto de **teste** só aparece na avaliação final (notebooks 04/05).
- Seleção de features e pré-processamento são ajustados só com dados de treino, dentro de cada fold.
- Cada execução carrega o mesmo `cycle_id` e `data_version`, do início ao fim.
- Os valores de `selecao`, `modelos`, `promocao` e `mlflow` em `config.yaml` são **propostas iniciais**. O dono da feature pode ajustá-los antes do primeiro ciclo oficial, registrando a mudança num commit com a tag da feature.

## Como o histórico e a avaliação individual ficam no git

O histórico de desenvolvimento é o próprio **git**. Cada pessoa faz commits com a própria identidade neste mesmo repositório.

**Convenção de commit:** a mensagem começa com a tag da feature, `[F3] descrição curta`. Para mais de uma, `[F3][F4]` ou `[F3,F4]`. Commits de documentação geral usam `[docs]`, de ajuste de equipe `[equipe]`, de correção `[fix]`.

`scripts/gerar_historico.py` lê as tags e `config/equipe.yaml` e monta o `HISTORICO_DESENVOLVIMENTO.md`, com:
- resumo por integrante;
- **visão por feature**: dono, commits do dono, commits de apoio e se a feature já tem commits do dono;
- linha do tempo completa.

**Fluxo recomendado**
1. `git clone` e uma branch por feature: `git switch -c f3-selecao`.
2. Commits `[F3] ...` com a **sua** identidade (`git config user.name` igual ao nome em `equipe.yaml`).
3. Pull Request para a `main` (há um template em `.github/pull_request_template.md`), integrando com **"Create a merge commit"** ou **"Rebase and merge"**. Evitem *squash*, que apaga os commits individuais.
4. Antes da entrega: `python scripts/gerar_historico.py` e commitar o arquivo gerado.

**Quem já desenvolveu em outro repositório** pode trazer o histórico preservando os commits originais:

```bash
git remote add colega https://github.com/USUARIO/REPO.git
git fetch colega
git merge --allow-unrelated-histories colega/main -m "[historico] incorpora trabalho de USUARIO"
```

**Identidades duplicadas** (mesma pessoa com dois e-mails) são unificadas no arquivo `.mailmap`.

**O que não fazer:** escrever commits em nome de outra pessoa, ou editar datas e autores para simular participação. O histórico serve de evidência de autoria e o professor pode conferi-lo. A tag `[Fn]` indica onde o trabalho foi feito, mas quem avalia é o conteúdo da feature, não a contagem de commits.
