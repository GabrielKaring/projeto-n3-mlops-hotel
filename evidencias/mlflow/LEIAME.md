# Evidências do MLflow (critério 6)

Execução local em 19/11/2026: notebook 01 → notebook 02, liberação do lote `lote_2017-01` (`python -m src.ingestao --liberar 1`) e de novo notebook 01 → notebook 02. O histórico foi lido de volta do MLflow com `python -m src.rastreio`.

| Arquivo | Conteúdo |
|---|---|
| `historico_runs.csv` | Os dois runs exportados do MLflow, com tags e métricas |
| `02_executado_ciclo1.ipynb` | Notebook 02 executado no ciclo 1, com a seção F6 |
| `02_executado_ciclo2.ipynb` | Notebook 02 executado no ciclo 2. A última tabela mostra os **dois** ciclos recuperados do MLflow |

## Runs registrados

Experimento `n3-hotel-cancelamento`, etapa `02_votacao`.

| Ciclo | `cycle_id` | `data_version` | `run_id` | Candidatas | Eleitas | Votos 3 / 2 / 1 / 0 |
|---|---|---|---|---|---|---|
| 1 | `20261119T182807Z-local` | `dv018-f95c3940` | `0aa32b5f87d044dfaa4fbef64b94fbeb` | 76 | 17 | 4 / 13 / 23 / 36 |
| 2 | `20261119T182922Z-local` | `dv019-aa1a1287` | `06391ee83b804ecb9d90ffee8b899e55` | 77 | 18 | 5 / 13 / 20 / 39 |

Cada run tem 16 parâmetros (seção `selecao` do `config.yaml`, seed, folds, tamanho do dev) e estes artefatos:

```
contexto/ciclo_atual.json
features.json
selecao/rankings_<cycle_id>.csv
selecao/rankings_por_fold_<cycle_id>.csv
votacao/votos_<cycle_id>.csv
votacao/features_eleitas.json
```

**O que mudou entre os ciclos:** o lote de jan/2017 trouxe uma categoria nova de agente (`agent_142`), então há 77 candidatas em vez de 76. A feature `agent_9` passou a ter votos suficientes e entrou no conjunto eleito, que foi de 17 para 18. O ciclo 1 reproduz exatamente a evidência da F4 (`../votacao/votos_20261105T123115Z-local.csv`): 17 eleitas e a mesma distribuição de votos.

## Como reproduzir e consultar

```bash
python -m src.rastreio
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db
```

## Limitações

- Os ciclos rodaram **localmente**: a pasta `mlflow_estado/` não está no repositório. A evidência oficial precisa vir de ciclos agendados no GitHub Actions, depois que a F7 aplicar a persistência proposta em `../../docs/f6_mlflow.md`.
- A tag `git_commit` dos dois runs aponta para `57525a6`, mas o código da F6 ainda não estava commitado nessa hora. Nas execuções do Actions o commit é sempre o código que rodou.
- Ainda não há runs das etapas 03 e 04, que dependem da F5 e da F8.
- Falta a captura de tela da interface do MLflow com os dois ciclos, para o relatório.
