# Roteiro da apresentação (30/11/2026)

Apoio para a socialização: quem fala em cada slide, o que mostrar na interface, o que preparar antes e as perguntas mais prováveis. Os números vêm do [relatório](../reports/relatorio_n3.md) e das evidências do Actions em `evidencias/actions/`.

## Quem fala e o que mostra

A nota da socialização é individual. Apresentam quatro integrantes, com 3 ou 4 slides cada, seguindo a ordem do pipeline; quem fala por uma parte feita por outro integrante cita o autor. A coluna "Na interface" diz qual página abrir se a professora pedir para ver funcionando (`streamlit run app/streamlit_app.py`).

| Quem | Slides | Na interface |
|---|---|---|
| José | 1 Capa · 2 Quem fez o quê · 6 Fluxo automático e MLflow · 14 O que entregamos | O projeto; José pode abrir `mlflow ui` ou rodar `python -m src.rastreio` |
| Gabriel Karing | 3 O problema · 4 Os dados · 5 O que separa quem cancela · 7 Votação das variáveis | O projeto; Os dados; Como o modelo é escolhido ("Detalhes técnicos") |
| Gabriel Wendorff | 8 Competição A × B · 10 Resultados · 11 Onde o modelo erra | Como o modelo é escolhido ("Detalhes técnicos"); O modelo acertou? (com "E se o corte fosse outro?") |
| Diego | 9 Promoção · 12 Alerta de drift (feito por Guilherme e Lucas) · 13 Demonstração | Como o modelo é escolhido; Alerta de mudança; Testar uma reserva |

Cada um deve conseguir explicar, sem o slide: o que a sua parte faz, por que foi feita assim e onde está a evidência no repositório (a feature e os arquivos estão em `config/equipe.yaml`).

## Checklist do dia

Na véspera, no computador que vai ser usado:

```bash
git pull                                              # main atualizada
source .venv/bin/activate                             # Windows: .venv\Scripts\activate
pip install -r requirements-app.txt -c constraints.txt
gh auth status                                        # o download do modelo usa o GitHub CLI
arquivo=$(python -c "import json; print(json.load(open('models/modelo_vigente_actions.json'))['vigente']['arquivo'])")
gh release download modelos --pattern "$arquivo" --dir models/registro --skip-existing
python -m src.prever --json exemplos/reservas_exemplo.json --manifesto models/modelo_vigente_actions.json
streamlit run app/streamlit_app.py
```

- O `src.prever` deve mostrar 82,3%, 100,0% e 1,6%. Se mostrar, o modelo está no lugar e a interface vai funcionar.
- Abra uma vez cada página da interface antes de apresentar: "Os dados", "O modelo acertou?" e "Alerta de mudança" leem todos os lotes na primeira vez (alguns segundos) e depois ficam em cache.
- Com o modelo baixado, a interface não precisa de internet.

**Plano B**, se a interface não abrir: rodar o `python -m src.prever` acima no terminal (é a mesma previsão). Se nem o Python funcionar, o slide 13 tem o print da interface e os resultados dos três exemplos.

## Perguntas prováveis

**Por que F1 macro e não acurácia?** 63% das reservas vieram e 37% cancelaram. Um modelo que dissesse sempre "vai vir" teria 63% de acurácia sem servir para nada. O F1 macro dá o mesmo peso às duas classes.

**Por que separar treino e teste pela data?** O cancelamento muda com o tempo (sobe em 2017). Na prática o modelo é treinado no passado e usado no futuro; o teste com as reservas mais recentes simula isso. Separar ao acaso misturaria meses futuros no treino.

**Usar o teste na decisão de promoção não é vazamento?** A escolha das features e entre A e B usa só o desenvolvimento, com validação cruzada temporal. O teste só aplica uma regra fixada antes (piso de 0,60 e ganho mínimo de 0,005), sem ajustar nada. A avaliação de verdade é a conferência das previsões com o mês seguinte, que não existia quando o modelo foi treinado (relatório, seção 7).

**Por que o modelo piorou em jul e ago/2017 (F1 0,68 e 0,70)?** O perfil das reservas mudou em 2017: a taxa de cancelamento ficou acima da de 2016 desde janeiro. O pipeline reagiu, porque o retreino passou a ganhar do modelo em uso e foi promovido três vezes seguidas (jun, jul e ago), mas não há um gatilho que retreine assim que o drift alerta. Está nas limitações.

**Por que o modelo perde tantos cancelamentos?** Com o corte em 50%, ele é conservador: no teste do dv026 deixou passar 38% dos cancelamentos, com 14% de alarme falso. O corte pode ser ajustado ao custo de cada erro para o hotel. Na interface, "O modelo acertou?" tem o controle ("E se o corte fosse outro?"): em ago/2017, baixar o corte de 50% para 40% reduz os cancelamentos perdidos de 51% para 26%, e os alarmes falsos sobem de 11% para 28%. Não mudamos o corte no pipeline porque ele foi fixado antes do treino.

**Por que "não reembolsável" cancela 99% das vezes?** É o que os dados mostram: das 14.587 reservas *Non Refund*, 99% constam como canceladas. O dataset não traz informação para explicar o motivo, então o modelo usa o padrão como ele está, e nós não inventamos uma causa. É também o atributo que os três métodos elegem por unanimidade em todos os ciclos.

**Por que o Random Forest ganhou?** Ele captura interações (por exemplo, antecedência longa e agência online juntas) que a regressão logística, linear, não captura. Venceu nos 8 ciclos, também na ROC-AUC (0,83 a 0,85 contra 0,81 a 0,84).

**Como vocês evitam vazamento de dados?** Saem as colunas que só existem depois da reserva (`reservation_status`, `assigned_room_type`, `booking_changes`, `country`), e tudo o que aprende com os dados (outliers, escala, categorias) fica dentro do Pipeline, ajustado só no treino. Há testes automáticos para isso.

**E se um passo do pipeline falhar?** O fluxo para e nada é promovido. A execução #3 foi uma falha simulada no treino: o 04 e a Release não rodaram e o modelo em uso continuou o mesmo.

**Onde está o histórico do MLflow, se o runner do GitHub é apagado?** Na branch `estado-pipeline`, gravada a cada execução e recuperada na seguinte. Para consultar: `python -m src.rastreio` ou `mlflow ui` (README, seção "MLflow").

**Por que tiraram o `agent` do alerta de drift?** É o código da agência, com centenas de valores, e a carteira de agências se renova todo ano: ele disparava alerta em todo mês. O PSI continua calculado e aparece no gráfico, mas não entra no status geral. A mudança de canal continua monitorada por `tem_agente`, `market_segment` e `distribution_channel`.

**Por que a melhoria mínima é 0,005?** Para não trocar de modelo por ruído: com alguns meses de teste, diferenças menores que isso podem ser acaso. Foi fixada antes do primeiro treino. No dv023 o novo ganhou por 0,0034 e o modelo em uso ficou.
