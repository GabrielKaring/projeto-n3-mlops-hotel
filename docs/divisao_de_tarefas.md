# Divisão de tarefas

Proposta inicial para 7 integrantes. Preencha os nomes e ajuste à vontade, mas registre a mudança em um commit.

> **Atenção:** o enunciado (`Trabalho_N3_Analise_Preditiva_MLOps.pdf`, capa) diz *"grupos de até 5 integrantes"*. Com 7 pessoas, confirmem com o professor antes da entrega.

## Quem faz o quê

| # | Integrante | GitHub | Critérios | Notebook / módulo | Status |
|---|---|---|---|---|---|
| 1 | Gabriel Henrique Karing | | **1, 2** e **9** (coordenação, README e relatório) | `01_etl`, `src/ingestao.py`, `src/preprocessamento.py`, `src/avaliacao.py` | 1 e 2 feitos; 9 em andamento |
| 2 | _nome_ | | **3, 4**: três métodos de seleção e votação | `02_votacao_features`, `src/features.py` | a fazer |
| 3 | _nome_ | | **5**: treino dos dois algoritmos | `03_treino_competicao`, `src/modelagem.py` | a fazer |
| 4 | _nome_ | | **6**: MLflow, persistência e rastreabilidade | `src/` (novo módulo de rastreio), `config.mlflow` | a fazer |
| 5 | _nome_ | | **7**: workflow único e agendamento | `.github/workflows/pipeline_mlops.yml` | esqueleto pronto |
| 6 | _nome_ | | **8**: eleição, promoção e previsões | `04_selecao_promocao`, `05_previsoes`, `src/promocao.py` | a fazer |
| 7 | _nome_ | | **10**: recurso adicional justificado | a definir | a fazer |

**Sugestão para o critério 10:** monitoramento de drift com alerta. O dataset tem mudança real de distribuição (a taxa de cancelamento sobe para ~43% em abr-jun/2017), então há um problema concreto para motivar o recurso. Ele deve **não** repetir os critérios 1 a 9: a validação de esquema que já existe em `validar_dados_brutos` faz parte do critério 2 e não conta.

## Dependências entre as partes

```
01_etl (pronto) ──> 02 (3, 4) ──> 03 (5) ──> 04 (8) ──> 05 (8)
        │                                       ▲
        └──────── 6 (MLflow) registra 02, 03, 04┘
7 (workflow) executa 01 → 04 (→ 05) e persiste o estado entre execuções
```

Quem faz a parte 5 e a parte 6 precisa conversar cedo: o runner do GitHub Actions é efêmero, então o histórico do MLflow e o `modelo_vigente.json` precisam de um lugar persistente.

## O que já está pronto e como usar

Contrato entre notebooks: o 01 grava `data/processed/ciclo_atual.json`, e os demais apenas leem.

```python
from src.config import carregar_config, carregar_contexto
from src.preprocessamento import carregar_particao, construir_preprocessador, lista_features
from src.avaliacao import criar_cv, metricas_classificacao, melhorou

cfg = carregar_config()
ctx = carregar_contexto(cfg)            # cycle_id, data_version, colunas, caminhos
dev = carregar_particao("dev", cfg)     # ordenado por data de chegada
pre = construir_preprocessador(cfg)     # NÃO ajustado: ajuste só no treino / dentro de cada fold
```

Regras que valem para todos (vêm do enunciado):
- A métrica (`f1_macro`) e o protocolo (`criar_cv`) estão definidos em `config/config.yaml`. Não troquem depois de ver resultados.
- O conjunto de **teste** só aparece na avaliação final (notebooks 04/05).
- Seleção de features e pré-processamento são ajustados só com dados de treino, dentro de cada fold.
- Cada execução carrega o mesmo `cycle_id` e `data_version`, do início ao fim.
- Os valores de `selecao`, `modelos`, `promocao` e `mlflow` em `config.yaml` são **propostas iniciais**. O dono de cada etapa pode ajustá-los antes do primeiro ciclo oficial.

## Como o histórico de todos fica junto

O histórico de desenvolvimento é o próprio **git**. Cada pessoa faz commits com a própria identidade neste mesmo repositório, e `scripts/gerar_historico.py` monta `HISTORICO_DESENVOLVIMENTO.md` a partir do `git log`, agrupando por integrante.

**Fluxo recomendado**
1. `git clone` do repositório e uma branch por tarefa: `git switch -c etapa-3-selecao`.
2. Mensagens de commit no formato `[etapa-3] descrição curta`.
3. Pull Request para a `main`, integrando com **"Create a merge commit"** ou **"Rebase and merge"**. Evitem *squash*, que apaga os commits individuais.
4. Antes da entrega: `python scripts/gerar_historico.py` e commitar o arquivo gerado.

**Quem já desenvolveu em outro repositório** pode trazer o histórico preservando os commits originais:

```bash
git remote add colega https://github.com/USUARIO/REPO.git
git fetch colega
git merge --allow-unrelated-histories colega/main -m "[historico] incorpora trabalho de USUARIO"
```

**Identidades duplicadas** (mesma pessoa com dois e-mails) são unificadas no arquivo `.mailmap`.

**O que não fazer:** escrever commits em nome de outra pessoa ou editar datas e autores para simular participação. O histórico serve de evidência de autoria e o professor pode conferi-lo.

## Checklist por critério (definição de pronto)

| Critério | Pronto quando |
|---|---|
| 3 | Os 3 métodos rodam no `02`, com ranking/importância de cada um registrados |
| 4 | Regra de votação definida antes (em `config.yaml`), tabela de votos e conjunto final salvos |
| 5 | Modelos A e B treinados sobre as mesmas features, mesma partição e mesma métrica, dois runs comparáveis |
| 6 | Dois ciclos diferentes recuperáveis no MLflow, com `run_id`, `cycle_id`, `data_version`, parâmetros, métricas e artefatos |
| 7 | Um único YAML, com `schedule` e execução agendada comprovada por log |
| 8 | Comparação automática, regra de promoção/manutenção registrada e previsões com o vigente e seu pré-processamento |
| 9 | README reproduzível, relatório PDF com a tabela de votação, a decisão de promoção e as limitações |
| 10 | Necessidade real, solução integrada ao fluxo e resultado demonstrado no relatório |
