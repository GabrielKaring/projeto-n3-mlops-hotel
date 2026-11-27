# Projeto N3 · Pipeline MLOps para previsão de cancelamento de reservas

O mesmo repositório é entregue para duas disciplinas:

- **Análise Preditiva**, Trabalho avaliativo N3 (Católica de Santa Catarina, Eng. de Software e Eng. de IA, Prof. Rodrigo Hagstrom);
- **Aprendizagem de Máquina**, projeto N3 da turma 8NA.

**Objetivo:** prever, no momento em que a reserva entra, se ela vai ser cancelada, para que o hotel controle overbooking, peça depósito nas reservas de risco e planeje a ocupação real.

Pipeline preditivo versionado e agendado que, a cada ciclo, incorpora dados novos, elege features por votação de três métodos, faz dois algoritmos competirem, registra tudo no MLflow e só promove um novo modelo se ele superar o vigente por uma regra definida antes.

**Problema:** prever se uma reserva de hotel será cancelada (classificação binária), usando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Detalhes e decisões em [`docs/etapa1_problema_dados_avaliacao.md`](docs/etapa1_problema_dados_avaliacao.md).

## Quem usaria

A **gestão de receita e de reservas do hotel**. No momento em que uma reserva entra, o modelo estima a chance de ela ser cancelada. Com isso a equipe pode:

- decidir quanto overbooking aceitar em cada data, sem ficar com quartos vazios nem recusar hóspede no balcão;
- pedir depósito, confirmar por contato ou oferecer tarifa não reembolsável às reservas de risco alto;
- prever a ocupação real (e não só a reservada) para escalar equipe e compras.

Quem opera o pipeline (a equipe de dados) acompanha os ciclos no MLflow e o alerta de drift, que avisa quando o perfil das reservas mudou e o modelo pode estar desatualizado.

## Principais resultados

Execuções do pipeline no GitHub Actions até 26/11/2026 (F1 macro no **teste do ciclo**, vencedor e vigente avaliados nos mesmos registros). Cópia das evidências em [`evidencias/actions/`](evidencias/actions/) e do manifesto em [`models/modelo_vigente_actions.json`](models/modelo_vigente_actions.json).

| Execução | Dados (lote novo) | A · F1 CV | B · F1 CV | Vencedor no teste | Vigente no teste | Decisão |
|---|---|---|---|---|---|---|
| [45090760506](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45090760506) | dv019 (jan/2017) | 0,747 | 0,790 | B · 0,731 | nenhum | promovido (1º ciclo) |
| [45099413735](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099413735) | dv020 (fev/2017) | 0,769 | 0,787 | B · 0,743 | 0,723 | promovido (+0,020) |
| [45099857641](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45099857641) | dv021 (mar/2017) | 0,731 | 0,767 | B · 0,737 | 0,742 | mantido |
| [45154548830](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45154548830) (`schedule`) | dv022 (abr/2017) | 0,711 | 0,744 | B · 0,749 | 0,752 | mantido |
| [45262196574](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45262196574) (`schedule`) | dv023 (mai/2017) | 0,707 | 0,758 | B · 0,755 | 0,751 | mantido (+0,003) |
| [45474058813](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45474058813) (`schedule`) | dv024 (jun/2017) | 0,695 | 0,753 | B · 0,755 | 0,749 | promovido (+0,006) |
| [45679932790](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45679932790) (`schedule`) | dv025 (jul/2017) | 0,695 | 0,764 | B · 0,747 | 0,735 | promovido (+0,012) |
| [45878188588](https://github.com/GabrielKaring/projeto-n3-mlops-hotel/actions/runs/45878188588) (`schedule`) | dv026 (ago/2017) | 0,701 | 0,761 | B · 0,742 | 0,735 | promovido (+0,007) |

- **O Random Forest (B) venceu a regressão logística (A) em todos os ciclos**, no CV e na ROC-AUC (~0,85 contra ~0,83). No teste, o F1 macro ficou entre **0,73 e 0,76**.
- **A regra de promoção funcionou nos dois sentidos:** promoveu quando o ganho passou de 0,005 e manteve o vigente quando o candidato não era melhor (no dv023 ganhou por 0,003 e mesmo assim ficou o vigente). O vigente atual é o B do ciclo dv026.
- **O agendamento roda sozinho:** as 5 últimas execuções (45154548830 a 45878188588) vieram do `schedule`, uma por dia, com todos os passos verdes. A falha simulada (45091260105) parou o fluxo antes da promoção.
- **Previsões conferidas com o que aconteceu:** cada ciclo previu o mês seguinte, ainda fora do pipeline. Comparando com o desfecho real, o F1 macro ficou entre 0,716 e 0,765 de fev a jun/2017 e caiu para 0,680 e 0,698 em jul e ago/2017, quando o retreino passou a ser promovido seguidamente.
- **Onde o modelo erra:** com o vigente atual no teste do ciclo dv026 (24.018 reservas, análise de erros do Actions em [`evidencias/actions/promocao/`](evidencias/actions/promocao/)), o erro total é de 24%. O erro que mais aparece é deixar passar cancelamento: 38% das reservas canceladas não foram previstas, contra 14% de alarme falso entre as que vieram. Quase todos os erros são de reservas sem depósito, e 75% deles vêm de agências online (*Online TA*); a faixa de 30 a 89 dias de antecedência tem a maior taxa de erro (32%). Detalhes e matriz de confusão na seção 7 do relatório.
- **Drift:** a taxa de cancelamento de 2017 sobe em relação ao mesmo mês de 2016 (jan/2017: +9,2 p.p.; abr/2017: +5,5 p.p.), e o alerta avisou. Em 23/11 o `agent` saiu do status geral do alerta, porque disparava em todo lote ([`docs/f10_drift.md`](docs/f10_drift.md)).

Relatório completo: [`reports/relatorio_n3.pdf`](reports/relatorio_n3.pdf).

## Status por feature

Cada feature é um critério do enunciado e tem um dono, ou dois quando dividida (1 a 4 features por pessoa), definido em [`config/equipe.yaml`](config/equipe.yaml).

| Feature | Critério | Dono | Status |
|---|---|---|---|
| F1 | 1 · Problema, dados e avaliação | Gabriel Karing | ✅ pronto |
| F2 | 2 · ETL e engenharia de atributos | Gabriel Karing | ✅ pronto (testes + 2 ciclos de evidência) |
| F3 | 3 · Três métodos de seleção | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/selecao/`) |
| F4 | 4 · Votação e eleição das features | Gabriel Karing | ✅ pronto (testes + evidência em `evidencias/votacao/`) |
| F5 | 5 · Dois algoritmos disputam | Gabriel Wendorff | ✅ pronto (testes + evidência em `evidencias/competicao/`) |
| F6 | 6 · MLflow e rastreabilidade | José | ✅ pronto (histórico persistido entre execuções na branch `estado-pipeline`) |
| F7 | 7 · Workflow único e agendamento | Diego | ✅ pronto (`schedule` + `workflow_dispatch`, ver [`docs/f7_workflow.md`](docs/f7_workflow.md)) |
| F8 | 8 · Eleição, promoção e previsões | Diego | ✅ pronto (testes + decisões em `models/modelo_vigente.json`, gerado a cada ciclo; o do Actions está copiado em [`models/modelo_vigente_actions.json`](models/modelo_vigente_actions.json), ver [`models/LEIAME.md`](models/LEIAME.md)) |
| F9 | 9 · Repositório, relatório e evidências | Vinicius | ✅ pronto ([`reports/relatorio_n3.pdf`](reports/relatorio_n3.pdf)) |
| F10 | 10 · Recurso adicional (drift com alerta) | Lucas e Guilherme | ✅ pronto (passo do workflow logo depois do ETL, ver [`docs/f10_drift.md`](docs/f10_drift.md)) |

Definição de pronto, evidência e teste de cada feature: [`docs/divisao_de_tarefas.md`](docs/divisao_de_tarefas.md). Commits seguem `[Fn] descrição`; o [`HISTORICO_DESENVOLVIMENTO.md`](HISTORICO_DESENVOLVIMENTO.md) mostra, por feature, quem fez o quê.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
```

O `constraints.txt` fixa as versões em que os testes e o pipeline passaram (Python 3.12, pandas 2.3.3, scikit-learn 1.9.1, mlflow 3.18.0, numpy 2.5.5), geradas com `pip freeze`. O workflow instala do mesmo jeito, então uma versão nova de alguma biblioteca não muda o resultado nem quebra o modelo salvo (o `joblib` do scikit-learn depende da versão). O `requirements.txt` continua com as faixas aceitas; o pandas está limitado a `<3` porque o mlflow 3.x exige isso.

## Execução local

```bash
python -m pytest -q                      # testes (ingestão, ETL, seleção de features, votação, ausência de vazamento, avaliação, MLflow, divisão da equipe, histórico)
python -m src.ingestao                   # registra lotes novos em data/raw e mostra data_version
python -m src.ingestao --liberar 1       # simula a chegada do próximo lote mensal (data/incoming -> data/raw)
python -m src.drift --historico          # drift do lote mais recente contra a referência (F10), saídas em evidencias/drift/ e run 01_drift no MLflow
python -m src.prever --json exemplos/reserva_exemplo.json   # demonstração: uma reserva nova (precisa de um vigente; num clone novo, ver 'Demonstração de uso')
```

Um ciclo completo, na mesma ordem do workflow (os notebooks executados vão para `evidencias/execucao/`):

```bash
python -m src.ingestao --liberar 1
bash scripts/executar_notebook.sh 01_etl
python -m src.drift --historico
bash scripts/executar_notebook.sh 02_votacao_features
bash scripts/executar_notebook.sh 03_treino_competicao
bash scripts/executar_notebook.sh 04_selecao_promocao
bash scripts/executar_notebook.sh 05_previsoes
python -m src.rastreio
```

No Windows, rode o `bash` pelo Git Bash ou WSL. Leva uns 2 minutos por ciclo.

`--liberar` move um arquivo de lugar. Para voltar ao estado inicial: `git restore data && git clean -fd data/raw`.

Os dados já estão no repositório, divididos em lotes mensais. Para regerá-los a partir do CSV original do Kaggle: `python scripts/preparar_lotes.py hotel_bookings.csv`.

## Demonstração de uso

O uso final do modelo é responder, para uma reserva que acabou de entrar, qual a chance de ela ser cancelada. A entrada é um JSON com os campos do dataset conhecidos no ato da reserva, **sem** o desfecho (`is_canceled`) e sem as colunas que só existem depois (`assigned_room_type`, `booking_changes`, `reservation_status`, `reservation_status_date`). `country`, `agent`, `company` e `children` são opcionais. Exemplos: [`exemplos/reserva_exemplo.json`](exemplos/reserva_exemplo.json) (uma reserva) e [`exemplos/reservas_exemplo.json`](exemplos/reservas_exemplo.json) (três).

Num clone novo ainda não há vigente local. Para usar o do Actions, baixe o arquivo dele da Release `modelos` (precisa do [GitHub CLI](https://cli.github.com/) autenticado) e aponte o manifesto copiado:

```bash
arquivo=$(python -c "import json; print(json.load(open('models/modelo_vigente_actions.json'))['vigente']['arquivo'])")
gh release download modelos --pattern "$arquivo" --dir models/registro
python -m src.prever --json exemplos/reservas_exemplo.json --manifesto models/modelo_vigente_actions.json
```

Depois de rodar um ciclo local (seção anterior), o vigente local é usado sem `--manifesto`: `python -m src.prever --json exemplos/reservas_exemplo.json`.

```
Reserva 1 (City Hotel, chegada 15/09/2017): chance de cancelamento 82,3% -> risco alto
Reserva 2 (City Hotel, chegada 12/10/2017): chance de cancelamento 100,0% -> risco alto
Reserva 3 (Resort Hotel, chegada 22/09/2017): chance de cancelamento 1,6% -> risco baixo
Limiar: 50% (a partir dele, risco alto; o mesmo corte da avaliação do modelo)
Modelo vigente: B do ciclo 20261126T184446Z-gha45878188588-1
```

A saída acima é do vigente do Actions (B do ciclo dv026). A reserva passa pelo mesmo ETL do treino, no modo inferência, e pelo vigente, carregado como no notebook 05 (`models/modelo_vigente.json` + o `Pipeline` completo em `models/registro/`). Em Python: `from src.prever import prever_reserva`.

Entradas inválidas são recusadas com uma mensagem, sem traceback: campo obrigatório vazio, texto num campo numérico (`"adr": "cento e dez"`; um número entre aspas, como `"110.5"`, é aceito), data que não existe (31 de setembro) e valor desconhecido em `hotel` ou `deposit_type` (`"hotel": "Hotel Central"`). Valor desconhecido nas outras categóricas (`meal`, `market_segment`, `customer_type` etc.) não interrompe, mas a reserva sai com um `AVISO`, porque o modelo não viu esse valor no treino. Os valores aceitos estão em `inferencia` no `config/config.yaml`.

Mais detalhes sobre os manifestos em [`models/LEIAME.md`](models/LEIAME.md). O notebook 05 também faz essa demonstração em todo ciclo e grava `evidencias/previsoes/demonstracao_<cycle_id>.csv`.

## Interface de demonstração (Streamlit)

Uma interface para apresentar o projeto funcionando, sem abrir notebooks:

```bash
pip install -r requirements-app.txt -c constraints.txt
gh release download modelos --pattern "$(python -c "import json; print(json.load(open('models/modelo_vigente_actions.json'))['vigente']['arquivo'])")" --dir models/registro
streamlit run app/streamlit_app.py
```

O segundo comando baixa o vigente do Actions (a interface também tem um botão para isso na barra lateral). Páginas:

| Página | O que mostra |
|---|---|
| O projeto | O problema, os números principais, como funciona o ciclo e quem usa |
| Os dados | Cancelamento por depósito, antecedência e pedidos especiais, e ao longo do tempo |
| Testar uma reserva | Três exemplos prontos e campos editáveis: a chance de cancelamento pelo vigente (`src.prever`) é recalculada na hora, com o gráfico "e se mudasse a antecedência?" |
| O modelo acertou? | As previsões que cada ciclo do Actions fez para o mês seguinte, conferidas com o que aconteceu, e a simulação de outro corte de decisão |
| Como o modelo é escolhido | Novo modelo × modelo em uso em cada mês e a decisão; a competição A × B e a votação das variáveis ficam em "Detalhes técnicos" |
| Alerta de mudança | Cancelamento de cada mês contra o mesmo mês do ano anterior, com os alertas; o PSI por variável fica em "Detalhes técnicos" |

Cada página mostra poucos números, um gráfico e uma frase de conclusão; o que é técnico fica recolhido em "Detalhes técnicos". O tema claro e o modo headless vêm de `.streamlit/config.toml`.

A interface só lê: não move lotes, não treina e não grava nada versionado. Para a apresentação, o [`docs/roteiro_apresentacao.md`](docs/roteiro_apresentacao.md) tem quem fala em cada slide, o checklist do dia com o plano B e as perguntas prováveis.

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     métrica, partição, ETL, votação, modelos, promoção, MLflow
config/equipe.yaml                     dono de cada feature (F1..F10), validado por tests/test_equipe.py
requirements.txt, constraints.txt      dependências e as versões fixas em que tudo foi testado
data/raw/                              lotes incorporados (*.csv.gz) + manifest.json
data/incoming/                         lotes futuros, liberados um a um para simular dados novos
data/processed/                        saídas intermediárias do ciclo (não versionadas)
docs/                                  decisões de cada feature, divisão de tarefas e gates
notebooks/01_etl.ipynb ... 05_previsoes.ipynb
src/                                   ingestao, preprocessamento, avaliacao, features, votacao, modelagem, rastreio,
                                       promocao, analise_erros, drift, prever (demonstração de uso)
tests/                                 pytest
exemplos/                              reservas novas, sem rótulo, para a demonstração de uso
models/modelo_vigente.json             vigente e histórico de decisões (gerado pelo notebook 04; oficial na branch estado-pipeline)
models/modelo_vigente_actions.json     cópia do manifesto do Actions (ver models/LEIAME.md)
models/registro/                       vencedor de cada ciclo (no Actions, fica na Release `modelos`)
mlflow_estado/                         histórico do MLflow: mlflow.db + artefatos/ (não versionado; ver docs/f6_mlflow.md)
evidencias/                            diagnósticos, notebooks executados, drift, decisões, análise de erros e previsões
evidencias/actions/                    cópia das evidências das execuções no GitHub Actions
reports/relatorio_n3.pdf               relatório final
app/streamlit_app.py                   interface de demonstração (Streamlit)
scripts/                               preparar_lotes.py, gerar_historico.py, executar_notebook.sh, gerar_relatorio.py,
                                       copiar_evidencias_actions.py
```

## Contrato entre notebooks

O notebook 01 grava `data/processed/ciclo_atual.json` (com `cycle_id`, `data_version`, caminhos e colunas), `dev.parquet` e `teste.parquet`. Os demais apenas leem, via `src.config.carregar_contexto`. O conjunto de teste não entra no treino nem na seleção (features e A × B usam só o `dev`, com validação cruzada temporal): ele só é usado no notebook 04, na comparação final vencedor × vigente, com a regra fixada antes no `config.yaml`, e na análise de erros. A avaliação de produção é a conferência das previsões do lote futuro no notebook 05 (relatório, seção 7). Cada ciclo carrega o mesmo `cycle_id` e `data_version` do começo ao fim.

## MLflow

Cada notebook registra seus runs no experimento `n3-hotel-cancelamento`, sempre com as tags `cycle_id` e `data_version`. Por ciclo são 5 runs: `01_drift`, `02_votacao`, `03_modelo_A`, `03_modelo_B` e `04_promocao` (este com a decisão, a matriz de confusão e a análise de erros). O histórico fica em `mlflow_estado/` (SQLite + artefatos). Detalhes em [`docs/f6_mlflow.md`](docs/f6_mlflow.md).

```bash
python -m src.rastreio                                           # lista ciclos e runs; exporta evidencias/mlflow/historico_runs.csv
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db  # interface em http://127.0.0.1:5000
```

## Execução agendada

O workflow [`.github/workflows/pipeline_mlops.yml`](.github/workflows/pipeline_mlops.yml) roda um ciclo por execução: recupera o estado, libera 1 lote novo, roda o notebook 01, o drift e os notebooks 02 a 05, e guarda o estado de novo. Dispara pelo `schedule` (todo dia às 06:00 de Brasília) ou manualmente em *Actions → pipeline-mlops → Run workflow*.

Como o runner é apagado no fim, o estado fica fora dele:

- branch `estado-pipeline`: histórico do MLflow, manifesto dos lotes, modelo vigente e previsões;
- Release `modelos`: o vencedor de cada ciclo (nada é apagado ou substituído);
- artefato `evidencias-<run_id>` de cada execução: evidências do ciclo.

Para abrir esse estado na sua máquina e consultar o MLflow:

```bash
git fetch origin estado-pipeline
git archive origin/estado-pipeline | tar -xf -
python -m src.rastreio
mlflow ui --backend-store-uri sqlite:///mlflow_estado/mlflow.db
```

Isso sobrescreve o `data/raw/manifest.json` local. Para voltar: `git checkout -- data/raw/manifest.json`. Detalhes, falha simulada e limitações em [`docs/f7_workflow.md`](docs/f7_workflow.md).

## Previsões

O notebook 05 carrega o vigente (Pipeline completo: pré-processamento, features eleitas e algoritmo) e prevê o próximo lote de `data/incoming/`, que ainda não participou da escolha do modelo. As previsões ficam em `evidencias/previsoes/previsoes_<cycle_id>.csv`, com o ciclo e o modelo que previu. Quando esse lote chega no ciclo seguinte, o próprio notebook confere as previsões com o rótulo real.

O notebook 04 também grava a matriz de confusão e a análise de erros do modelo em uso no teste do ciclo (por hotel, segmento, tipo de depósito, tipo de cliente, canal e faixa de antecedência) em `evidencias/promocao/`.

**Quando os lotes acabam.** Com o `schedule` diário, o último lote de 2017 entrou em 26/11. Daí em diante o notebook 05 avisa, mostra as últimas previsões feitas e mantém a demonstração; o agendamento não foi alterado. Detalhes em [`docs/f7_workflow.md`](docs/f7_workflow.md).

Para prever com o modelo vigente localmente, rode o ciclo completo acima (ou abra o estado do Actions e baixe o vigente com `gh release download modelos --dir models/registro`) e depois `bash scripts/executar_notebook.sh 05_previsoes`.

## Dados e licença

Antonio, Almeida & Nunes (2019). *Hotel booking demand datasets*. Data in Brief 22:41-49. DOI 10.1016/j.dib.2018.11.126. Licença **CC BY** (artigo de dados original). Nenhum segredo ou dado restrito é versionado.

## Referências

- ANTONIO, N.; ALMEIDA, A.; NUNES, L. Hotel booking demand datasets. *Data in Brief*, v. 22, p. 41-49, 2019. DOI: [10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126).
- MOSTIPAK, J. *Hotel booking demand* (versão do dataset no Kaggle). Disponível em: <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825-2830, 2011. Documentação: <https://scikit-learn.org>.
- MLflow: plataforma aberta para o ciclo de vida de modelos de machine learning. Documentação: <https://mlflow.org/docs/latest/>.
- SIDDIQI, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Hoboken: Wiley, 2006. (Limiares usuais do PSI usados no monitoramento de drift.)
- GitHub Actions: documentação de workflows e eventos `schedule` e `workflow_dispatch`. <https://docs.github.com/actions>.
