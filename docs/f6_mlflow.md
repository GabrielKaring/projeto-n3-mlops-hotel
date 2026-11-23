# F6 · MLflow e rastreabilidade

Dono: José. Código em [`src/rastreio.py`](../src/rastreio.py), testes em [`tests/test_rastreio.py`](../tests/test_rastreio.py), evidência em [`evidencias/mlflow/`](../evidencias/mlflow/).

## O que é registrado

Um **experimento** (`n3-hotel-cancelamento`, em `config.yaml`) e um **run por etapa de cada ciclo**. Todo run leva as mesmas tags, o que permite juntar as etapas de um ciclo e comparar ciclos:

| Tag | Exemplo | Para quê |
|---|---|---|
| `cycle_id` | `20261119T181200Z-local` | Identifica a execução do pipeline (vem do notebook 01) |
| `data_version` | `dv019-1a2b3c4d` | Identifica os lotes de dados usados |
| `etapa` | `01_drift`, `02_votacao`, `03_modelo_A`, `04_promocao` | Em que passo do ciclo o run foi aberto |
| `origem` | `local` ou `github-actions` | Onde rodou |
| `git_commit` | `57525a6...` | Versão do código |
| `decisao` | `promovido` ou `mantido` | Só no run da promoção (F8) |
| `drift_status` | `ok`, `atencao`, `alerta`, ... | Só no run do drift (F10) |

Cada run também guarda `contexto/ciclo_atual.json`, o contexto completo do ciclo.

| Etapa | Feature | Parâmetros | Métricas | Artefatos |
|---|---|---|---|---|
| `01_drift` | F10 (`src/drift.py`, passo logo depois do notebook 01) | lote avaliado, referência, limiares, features informativas | `psi_<feature>`, variação da taxa de cancelamento, nº de features em alerta | arquivos de `evidencias/drift/` do ciclo |
| `02_votacao` | F3 + F4 (registro feito pela F6) | seção `selecao` inteira, seed, folds | nº de candidatas, nº de eleitas, features com 0/1/2/3 votos | rankings, rankings por fold, tabela de votos, `features_eleitas.json` |
| `03_modelo_A/B` | F5 | `registrar_parametros(...)` | `registrar_metricas(...)` | `registrar_features`, `registrar_modelo` |
| `04_promocao` | F8 | regra de promoção | métricas do vencedor e do vigente; matriz de confusão do modelo em uso (`teste_em_uso_*`) | `registrar_decisao(...)`, análise de erros (`analise_erros/`) |

As linhas de F5 e F8 são o contrato: as funções existem e têm teste, mas a chamada fica nos notebooks 03 e 04, dos respectivos donos.

## Onde fica guardado

Tudo numa pasta só, `mlflow_estado/` (configurável em `config.yaml`, `mlflow.dir_estado`):

```
mlflow_estado/
  mlflow.db       SQLite com experimentos, runs, tags, parâmetros e métricas
  artefatos/      arquivos de cada run
```

Copiar a pasta é copiar o histórico inteiro. Ela não é versionada na `main` (está no `.gitignore`).

**Por que SQLite e não a pasta `mlruns/`:** um único arquivo de banco é mais fácil de guardar e restaurar, e as consultas por tag (`historico_ciclos`) são feitas pelo banco em vez de varrer milhares de arquivos.

**Caminho dos artefatos:** o MLflow grava no banco o caminho absoluto da pasta de artefatos. Se o histórico é criado no runner (`/home/runner/...`) e aberto no Windows (`C:/...`), os artefatos não seriam encontrados. Por isso `configurar_mlflow` chama `_relocar_artefatos`, que troca o caminho antigo pelo atual antes de abrir o banco. O teste `test_historico_continua_valido_quando_a_pasta_muda_de_lugar` simula exatamente isso.

## Como consultar

```bash
python -m src.rastreio                                           # lista ciclos e runs e exporta evidencias/mlflow/historico_runs.csv
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db  # interface em http://127.0.0.1:5000
```

No código: `historico_ciclos(cfg)` devolve todos os runs; `historico_ciclos(cfg, cycle_id="...")` devolve só os de um ciclo.

## Persistência entre execuções do Actions (a combinar com F7)

O runner do GitHub Actions é apagado no fim de cada execução. Sem guardar `mlflow_estado/`, todo ciclo agendado começaria com o histórico vazio.

| Opção | Dura | Prós | Contras |
|---|---|---|---|
| **A. Branch `estado-pipeline`** (proposta) | Para sempre | Versionado, o professor consegue ver, sem serviço externo | O workflow precisa de `contents: write`; o banco cresce no git. **O modelo B da F5 passa de 50 MB**, e o GitHub recusa arquivos acima de 100 MB: os `.joblib` não devem ir para a branch (guardar só o banco e os artefatos pequenos, ou comprimir o modelo) |
| B. Artefato do Actions | 90 dias | Não mexe no repositório | Achar o artefato da execução anterior exige `gh run download` com filtros; expira |
| C. `actions/cache` | Pode ser apagado após 7 dias sem uso | Simples | Não é garantido. Não serve como histórico |

**Proposta para o Diego (opção A).** Ainda **não testado no Actions**: é um ponto de partida para o `pipeline_mlops.yml`. A mesma branch pode guardar também o manifesto dos lotes e `models/modelo_vigente.json`, que são da F7 e da F8.

```yaml
permissions:
  contents: write            # para gravar na branch estado-pipeline

concurrency:                 # o SQLite aceita um escritor por vez: nunca dois ciclos juntos
  group: pipeline-mlops
  cancel-in-progress: false

# ...depois do checkout e da instalação:
      - name: Recuperar estado persistente
        id: restaurar
        run: |
          if git fetch --depth=1 origin estado-pipeline; then
            git restore --source=FETCH_HEAD --worktree -- mlflow_estado
          else
            echo "Primeira execução: sem estado anterior."
          fi

# ...notebooks 01 a 05...

      - name: Guardar estado persistente
        # always(): runs que falharam também são histórico. Mas só se a restauração deu certo,
        # senão um estado vazio sobrescreveria o histórico.
        if: always() && steps.restaurar.outcome == 'success'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          if git fetch --depth=1 origin estado-pipeline; then
            git worktree add /tmp/estado FETCH_HEAD
          else
            git worktree add --orphan -b estado-pipeline /tmp/estado
          fi
          rm -rf /tmp/estado/mlflow_estado && cp -r mlflow_estado /tmp/estado/
          cd /tmp/estado
          git add -A
          git commit -m "[estado] ciclo $(jq -r .cycle_id $GITHUB_WORKSPACE/data/processed/ciclo_atual.json)" || exit 0
          git push origin HEAD:estado-pipeline
```

Para abrir localmente o histórico do Actions:

```bash
git fetch origin estado-pipeline
git restore --source=origin/estado-pipeline --worktree -- mlflow_estado
python -m src.rastreio
```

## Limitações

- Até a F7 aplicar a persistência, o histórico existe só na máquina onde o pipeline rodou.
- SQLite não aceita duas execuções escrevendo ao mesmo tempo. O `concurrency` acima evita isso no Actions.
- A relocação dos caminhos só vale para experimentos criados por `src/rastreio.py`, cuja pasta de artefatos termina em `/artefatos`.
- Os dois ciclos de `evidencias/mlflow/` foram rodados localmente. A evidência oficial deve vir de ciclos agendados no Actions (gate GF).
