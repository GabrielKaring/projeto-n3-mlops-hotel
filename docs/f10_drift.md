# F10 · Recurso adicional: monitoramento de drift com alerta

Critério 10 do enunciado. Dono em [`config/equipe.yaml`](../config/equipe.yaml). Código em [`src/drift.py`](../src/drift.py), testes em [`tests/test_drift.py`](../tests/test_drift.py), regra na seção `drift` de [`config/config.yaml`](../config/config.yaml).

## Problema

A cada ciclo chega um lote mensal novo de reservas, e o modelo vigente foi treinado com o passado. Se o perfil das reservas ou a taxa de cancelamento mudam, o modelo pode piorar sem que ninguém perceba.

A mudança existe neste dataset. A taxa de cancelamento sobe para cerca de 43% em abr–jun/2017, contra 33–37% nos meses vizinhos ([Etapa 1](etapa1_problema_dados_avaliacao.md)). Esses lotes chegam em ciclos futuros (`data/incoming/`), então o pipeline vai passar por eles.

## Por que não repete os critérios 1 a 9

| Já existe | O que responde | O que o drift responde |
|---|---|---|
| `validar_dados_brutos` (F2) | O lote tem as colunas e os valores válidos? | O lote **parece** com o que o modelo viu? |
| Competição e promoção (F5, F8) | Qual modelo tem a melhor métrica no ciclo? | **Por que** a métrica mudou, e em quais variáveis? |
| MLflow (F6) | O que rodou e com quais parâmetros? | Registra outra coisa: a mudança dos dados entre lotes. |

O monitoramento só lê dados. Nada do que ele calcula entra na seleção de features, no treino ou na promoção. É um aviso para quem opera o modelo, e não muda nenhuma decisão automática.

## Regra (fixada antes de rodar nos dados reais)

Tudo está em `config.yaml`, seção `drift`. A regra foi commitada antes da primeira execução sobre os lotes reais.

- **Referência sazonal.** O lote novo é comparado com o mesmo mês do ano anterior (`lote_2017-04` × `lote_2016-04`). Reservas de hotel são sazonais: comparar abril com o ano inteiro acusaria "drift" todo verão. Sem o mesmo mês, a referência são os últimos 12 lotes. A referência nunca usa lotes posteriores ao avaliado.
- **PSI por feature** (Population Stability Index). Numéricas contínuas usam os decis da referência. Categóricas e numéricas com até 10 valores usam as categorias, e as raras (< 1% na referência) viram um grupo só, como no one-hot do modelo.
  - PSI < 0,10 = `ok`; 0,10 a 0,25 = `atencao`; ≥ 0,25 = `alerta`. São os limiares usuais da literatura de risco de crédito (Siddiqi, 2006).
- **Drift do alvo.** Variação absoluta da taxa de cancelamento: ≥ 3 p.p. = `atencao`, ≥ 5 p.p. = `alerta`.
- **Colunas ignoradas.** Semana do ano, mês e dia da semana mudam por construção entre lotes mensais.
- **Amostra mínima.** Com menos de 500 linhas, o status é `amostra_insuficiente` e não há alerta.
- **Status geral.** É o mais grave entre as features e o alvo.

## Ponto de integração

Um passo no workflow único, logo depois do notebook 01. Ele precisa vir depois do 01 para usar o mesmo `cycle_id` e `data_version` do ciclo (`data/processed/ciclo_atual.json`):

```yaml
      - name: 01 - ETL
        run: bash scripts/executar_notebook.sh 01_etl

      - name: Monitoramento de drift (F10)
        run: python -m src.drift --historico

      - name: 02 - Seleção e votação de features
        ...
```

No Actions, o alerta aparece como `::warning` na página da execução e como tabela no resumo do job. Os arquivos ficam em `evidencias/drift/`, que o passo "Publicar evidências" já envia como artefato.

O passo já está no `.github/workflows/pipeline_mlops.yml`, entre o notebook 01 e o 02.

## Como rodar localmente

```bash
python -m pytest tests/test_drift.py -q
python -m src.drift                      # lote mais recente × referência
python -m src.drift --historico          # também todos os lotes incorporados, um por linha
python -m src.drift --lote lote_2016-08  # um lote específico
```

Para ver o alerta nos lotes de 2017, libere lotes novos (`python -m src.ingestao --liberar 1`), rode o notebook 01 e depois `python -m src.drift`. Se o `ciclo_atual.json` for de outra versão dos dados, o módulo para e pede o notebook 01. Sem esse arquivo, ele gera um `cycle_id` próprio.

## Saídas (`evidencias/drift/`)

| Arquivo | Conteúdo |
|---|---|
| `drift_<cycle_id>.json` | Status geral, drift do alvo, PSI e status por feature, referência, limiares, `cycle_id`, `data_version` |
| `drift_<cycle_id>.csv` / `.png` | Tabela e gráfico do PSI por feature |
| `historico_<cycle_id>.csv` / `.png` | Um lote por linha, cada um contra a sua referência (com `--historico`) |

## Limitações

- O PSI é univariado: não detecta mudança só na relação entre variáveis.
- Na operação real, o cancelamento só é conhecido depois da data de chegada. Aqui os lotes já trazem o alvo, então o drift do alvo aparece no mesmo ciclo.
- O monitoramento lê o lote mais recente, que cai na partição de teste do ciclo. Ele não ajusta nada nem altera decisões, então não há vazamento para a seleção, o treino ou a promoção.
- O alerta não dispara retreino. A decisão continua sendo da regra de promoção (F8).

## Situação

- [x] Regra e justificativa registradas (`config.yaml` e este documento).
- [x] `src/drift.py` e `tests/test_drift.py` escritos.
- [ ] `python -m pytest -q` executado e passando.
- [ ] Passo no workflow (depende da F7 na `main`).
- [ ] Execução real com os lotes de 2017, evidências em `evidencias/drift/` e `LEIAME.md`.
- [ ] Registro no MLflow (depende da F6 na `main`).
- [ ] Seção no relatório: problema, justificativa, solução, ponto de integração e resultado.
