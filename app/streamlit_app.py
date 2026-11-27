"""Interface de demonstração do pipeline MLOps (previsão de cancelamento de reservas).

Uso: ``streamlit run app/streamlit_app.py`` a partir da raiz do projeto.

Só lê: o manifesto do vigente, as evidências copiadas do GitHub Actions e os lotes de ``data/``. Não move
lotes, não treina e não grava nada no repositório (a única escrita possível é baixar o arquivo do vigente
da Release ``modelos`` para ``models/registro/``, que o git ignora). As previsões usam as mesmas funções do
pipeline (``src.prever`` e ``src.promocao``), então o que aparece aqui é o que o notebook 05 faria.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import promocao  # noqa: E402
from src.analise_erros import erros_por_grupo, faixa_lead_time, matriz_confusao, resumo_confusao  # noqa: E402
from src.avaliacao import metricas_classificacao  # noqa: E402
from src.config import carregar_config  # noqa: E402
from src.drift import monitorar_lote  # noqa: E402
from src.preprocessamento import limpar_e_engenhar  # noqa: E402
from src.prever import prever_reserva  # noqa: E402
from src.schema import MESES  # noqa: E402

MANIFESTO_ACTIONS = RAIZ / "models" / "modelo_vigente_actions.json"
MANIFESTO_LOCAL = RAIZ / "models" / "modelo_vigente.json"
EVID = RAIZ / "evidencias" / "actions"
MESES_NOME = {v: k for k, v in MESES.items()}
COR_A, COR_B = "#8c9eff", "#ff8a65"
NOMES_MODELO = {"A": "A · Regressão logística", "B": "B · Random Forest"}

st.set_page_config(page_title="N3 · Cancelamento de reservas", page_icon="🏨", layout="wide")


# --------------------------------------------------------------------------- utilitários


def pct(valor: float, casas: int = 1) -> str:
    return f"{valor:.{casas}%}".replace(".", ",")


def num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def dv_curto(data_version: str) -> str:
    return data_version.split("-")[0] if isinstance(data_version, str) else "?"


@st.cache_data
def cfg() -> dict:
    return carregar_config()


@st.cache_data
def ler_json(caminho: str) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def manifestos_disponiveis() -> dict[str, Path]:
    opcoes = {"Vigente do GitHub Actions": MANIFESTO_ACTIONS}
    if MANIFESTO_LOCAL.exists():
        opcoes["Vigente do ciclo local"] = MANIFESTO_LOCAL
    return opcoes


@st.cache_resource(show_spinner="Carregando o modelo vigente…")
def carregar_modelo(caminho_manifesto: str):
    manifesto = json.loads(Path(caminho_manifesto).read_text(encoding="utf-8"))
    return promocao.carregar_vigente(manifesto, RAIZ)


@st.cache_data(show_spinner="Lendo e limpando os lotes…")
def todos_os_lotes() -> pd.DataFrame:
    """Todos os lotes mensais (``data/raw`` + ``data/incoming``) com o mesmo ETL do pipeline, com rótulo."""
    c = cfg()
    arquivos = sorted((RAIZ / c["dados"]["dir_raw"]).glob("lote_*.csv.gz"))
    arquivos += sorted((RAIZ / c["dados"]["dir_incoming"]).glob("lote_*.csv.gz"))
    partes = []
    for arq in arquivos:
        bruto = pd.read_csv(arq)
        bruto.insert(0, "linha_origem", range(len(bruto)))
        bruto.insert(0, "lote_id", arq.name.removesuffix(".csv.gz"))
        partes.append(bruto)
    limpo, _ = limpar_e_engenhar(pd.concat(partes, ignore_index=True), c)
    return limpo


COLS_CONFERENCIA = ["is_canceled", "hotel", "market_segment", "deposit_type", "customer_type",
                     "distribution_channel", "lead_time", "adr"]


@st.cache_data(show_spinner="Lendo as previsões gravadas pelo Actions…")
def previsoes_actions() -> dict[str, Path]:
    """Lote previsto -> arquivo de previsões do ciclo que o previu (cada ciclo prevê o lote seguinte)."""
    saida = {}
    for arq in sorted((EVID / "previsoes").glob("previsoes_*.csv.gz")):
        lote = pd.read_csv(arq, usecols=["lote_id"], nrows=1).iloc[0, 0]
        saida[lote] = arq
    return saida


@st.cache_data(show_spinner="Conferindo as previsões com o desfecho real…")
def conferir_previsoes(arquivo: str) -> pd.DataFrame:
    """Previsões gravadas no Actions + rótulo real e atributos do lote (o mesmo cálculo do notebook 05)."""
    prev = pd.read_csv(arquivo)
    lote = prev["lote_id"].iloc[0]
    c = cfg()
    origem = next(p for d in (c["dados"]["dir_incoming"], c["dados"]["dir_raw"])
                  if (p := RAIZ / d / f"{lote}.csv.gz").exists())
    df, _ = promocao.preparar_lote(origem, c)
    return prev.merge(df[["lote_id", "linha_origem", *COLS_CONFERENCIA]], on=["lote_id", "linha_origem"])


# --------------------------------------------------------------------------- barra lateral

st.sidebar.title("🏨 Projeto N3")
st.sidebar.caption("Pipeline MLOps · previsão de cancelamento de reservas")

PAGINAS = [
    "Visão geral",
    "Exploração dos dados",
    "Simular uma reserva",
    "Previsões × realidade",
    "Ciclos e promoção",
    "Seleção de features",
    "Monitoramento de drift",
]
pagina = st.sidebar.radio("Navegação", PAGINAS, label_visibility="collapsed")

opcoes = manifestos_disponiveis()
nome_manifesto = st.sidebar.selectbox("Modelo usado nas previsões", list(opcoes), index=0)
caminho_manifesto = str(opcoes[nome_manifesto])
manifesto = ler_json(caminho_manifesto)
vigente = manifesto.get("vigente") or {}

modelo = None
erro_modelo = None
try:
    modelo, vigente = carregar_modelo(caminho_manifesto)
except FileNotFoundError as e:
    erro_modelo = str(e)

with st.sidebar.container(border=True):
    if vigente:
        st.markdown(f"**Vigente:** {NOMES_MODELO.get(vigente.get('modelo'), vigente.get('modelo'))}")
        st.caption(
            f"Ciclo `{vigente.get('cycle_id')}`  \n"
            f"Dados: `{vigente.get('data_version')}`  \n"
            f"F1 macro na promoção: **{num(vigente.get('valor_teste_na_promocao', 0))}**  \n"
            f"{len(vigente.get('features', []))} features eleitas"
        )
    if erro_modelo:
        st.error("Arquivo do modelo não encontrado em `models/registro/`.")
        if st.button("Baixar da Release `modelos`", width="stretch"):
            with st.spinner("Baixando com o GitHub CLI…"):
                r = subprocess.run(
                    ["gh", "release", "download", "modelos", "--pattern", vigente["arquivo"],
                     "--dir", str(RAIZ / "models" / "registro"), "--skip-existing"],
                    capture_output=True, text=True, cwd=RAIZ,
                )
            if r.returncode == 0:
                carregar_modelo.clear()
                st.rerun()
            else:
                st.code(r.stderr or r.stdout)


def exigir_modelo() -> None:
    if modelo is None:
        st.warning(
            "Esta página precisa do arquivo do modelo vigente. Use o botão na barra lateral "
            "ou rode, na raiz do projeto:"
        )
        st.code(f"gh release download modelos --pattern {vigente.get('arquivo')} --dir models/registro")
        st.stop()


# --------------------------------------------------------------------------- páginas


def pagina_visao_geral() -> None:
    st.title("Previsão de cancelamento de reservas de hotel")
    st.markdown(
        "Pipeline preditivo **versionado e agendado**: a cada ciclo incorpora um lote mensal novo, elege "
        "features por votação de três métodos, faz dois algoritmos competirem, registra tudo no **MLflow** "
        "e só promove um novo modelo se ele superar o vigente pela regra definida antes do treino."
    )

    hist = manifesto.get("historico", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclos decididos no Actions", len(hist))
    c2.metric("Promoções", sum(h["decisao"] == "promovido" for h in hist))
    c3.metric("F1 macro do vigente", num(vigente.get("valor_teste_na_promocao", 0)))
    c4.metric("Lotes mensais", f"{len(list((RAIZ / 'data/raw').glob('*.csv.gz')))} + "
                               f"{len(list((RAIZ / 'data/incoming').glob('*.csv.gz')))} futuros")

    st.subheader("O fluxo de um ciclo")
    etapas = [
        ("1 · Ingestão", "Lote novo entra em `data/raw`, gera `data_version` (hash dos lotes)."),
        ("2 · ETL", "Remove vazamento e colunas pós-reserva, cria atributos, partição temporal 80/20."),
        ("3 · Drift", "PSI por feature e taxa de cancelamento contra o mesmo mês do ano anterior."),
        ("4 · Votação", "SelectKBest, Random Forest e RFE votam; entra quem tiver ≥ 2 votos."),
        ("5 · Competição", "Regressão logística (A) × Random Forest (B) em CV cronológica."),
        ("6 · Promoção", "Vencedor substitui o vigente só se ganhar ≥ 0,005 de F1 no teste."),
        ("7 · Previsões", "O vigente prevê o próximo lote, que ainda não entrou no pipeline."),
    ]
    for linha in (etapas[:4], etapas[4:]):
        cols = st.columns(4)
        for col, (titulo, texto) in zip(cols, linha):
            with col.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(texto)

    st.subheader("Para quem serve")
    st.markdown(
        "- **Gestão de receita:** decidir quanto overbooking aceitar em cada data.\n"
        "- **Reservas:** pedir depósito ou confirmar por contato as reservas de risco alto.\n"
        "- **Operação:** prever a ocupação real para escalar equipe e compras.\n"
        "- **Equipe de dados:** acompanhar os ciclos no MLflow e o alerta de drift."
    )

    st.subheader("Divisão por feature")
    equipe = RAIZ / "config" / "equipe.yaml"
    if equipe.exists():
        import yaml

        dados = yaml.safe_load(equipe.read_text(encoding="utf-8"))
        linhas = []
        for fid, info in (dados.get("features") or {}).items():
            if isinstance(info, dict):
                donos = info.get("donos") or info.get("dono") or ""
                linhas.append({"Feature": fid, "Critério": info.get("titulo") or info.get("criterio", ""),
                               "Dono(s)": ", ".join(donos) if isinstance(donos, list) else donos})
        if linhas:
            st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.caption("Execute com `streamlit run app/streamlit_app.py`. A interface só lê; nada é gravado no repositório.")


def formulario_reserva() -> dict:
    c = cfg()["inferencia"]["dominio_categorias"]
    exemplos = json.loads((RAIZ / "exemplos" / "reservas_exemplo.json").read_text(encoding="utf-8"))
    rotulos = ["(preencher manualmente)"] + [
        f"Exemplo {i + 1}: {r['hotel']}, {r['arrival_date_day_of_month']}/{MESES[r['arrival_date_month']]}"
        f"/{r['arrival_date_year']}" for i, r in enumerate(exemplos)
    ]
    escolha = st.selectbox("Partir de um exemplo de `exemplos/reservas_exemplo.json`", rotulos)
    base = exemplos[rotulos.index(escolha) - 1] if escolha != rotulos[0] else json.loads(
        (RAIZ / "exemplos" / "reserva_exemplo.json").read_text(encoding="utf-8"))
    k = f"ex{rotulos.index(escolha)}"  # chave muda com o exemplo, então os campos são recarregados

    def idx(campo: str) -> int:
        return c[campo].index(base[campo]) if base.get(campo) in c[campo] else 0

    with st.form("reserva"):
        st.markdown("**Reserva e estadia**")
        a1, a2, a3, a4 = st.columns(4)
        hotel = a1.selectbox("Hotel", c["hotel"], index=idx("hotel"), key=f"hotel{k}")
        chegada = a2.date_input("Data de chegada", key=f"data{k}", value=date(
            int(base["arrival_date_year"]), MESES[base["arrival_date_month"]], int(base["arrival_date_day_of_month"])))
        lead = a3.number_input("Antecedência (dias)", 0, 800, int(base["lead_time"]), key=f"lead{k}")
        adr = a4.number_input("Diária média (adr)", 0.0, 1000.0, float(base["adr"]), step=5.0, key=f"adr{k}")
        b1, b2, b3, b4 = st.columns(4)
        fds = b1.number_input("Noites de fim de semana", 0, 20, int(base["stays_in_weekend_nights"]), key=f"fds{k}")
        semana = b2.number_input("Noites de semana", 0, 50, int(base["stays_in_week_nights"]), key=f"sem{k}")
        adultos = b3.number_input("Adultos", 0, 10, int(base["adults"]), key=f"ad{k}")
        criancas = b4.number_input("Crianças", 0, 10, int(base.get("children") or 0), key=f"cr{k}")

        st.markdown("**Canal e condições**")
        d1, d2, d3, d4 = st.columns(4)
        deposito = d1.selectbox("Depósito", c["deposit_type"], index=idx("deposit_type"), key=f"dep{k}")
        segmento = d2.selectbox("Segmento", c["market_segment"], index=idx("market_segment"), key=f"seg{k}")
        canal = d3.selectbox("Canal de distribuição", c["distribution_channel"],
                             index=idx("distribution_channel"), key=f"can{k}")
        cliente = d4.selectbox("Tipo de cliente", c["customer_type"], index=idx("customer_type"), key=f"cli{k}")
        e1, e2, e3, e4 = st.columns(4)
        refeicao = e1.selectbox("Refeição", c["meal"], index=idx("meal"), key=f"mea{k}")
        quarto = e2.selectbox("Quarto reservado", c["reserved_room_type"], index=idx("reserved_room_type"), key=f"qu{k}")
        pedidos = e3.number_input("Pedidos especiais", 0, 5, int(base["total_of_special_requests"]), key=f"ped{k}")
        vagas = e4.number_input("Vagas de garagem", 0, 3, int(base["required_car_parking_spaces"]), key=f"vag{k}")

        with st.expander("Histórico do hóspede e agência"):
            f1, f2, f3, f4 = st.columns(4)
            repetido = f1.checkbox("Hóspede recorrente", bool(base["is_repeated_guest"]), key=f"rep{k}")
            canc_prev = f2.number_input("Cancelamentos anteriores", 0, 30, int(base["previous_cancellations"]), key=f"cp{k}")
            nao_canc = f3.number_input("Reservas anteriores não canceladas", 0, 80,
                                       int(base["previous_bookings_not_canceled"]), key=f"nc{k}")
            espera = f4.number_input("Dias na lista de espera", 0, 400, int(base["days_in_waiting_list"]), key=f"esp{k}")
            g1, g2, g3 = st.columns(3)
            agente = g1.number_input("Código da agência (0 = sem agência)", 0, 600, int(base.get("agent") or 0), key=f"ag{k}")
            bebes = g2.number_input("Bebês", 0, 10, int(base["babies"]), key=f"beb{k}")
            pais = g3.text_input("País (ISO-3)", base.get("country") or "PRT", key=f"pa{k}")
        enviar = st.form_submit_button("Calcular chance de cancelamento", type="primary", width="stretch")

    reserva = {
        "hotel": hotel, "lead_time": int(lead),
        "arrival_date_year": chegada.year, "arrival_date_month": MESES_NOME[chegada.month],
        "arrival_date_week_number": int(chegada.isocalendar()[1]), "arrival_date_day_of_month": chegada.day,
        "stays_in_weekend_nights": int(fds), "stays_in_week_nights": int(semana),
        "adults": int(adultos), "children": int(criancas), "babies": int(bebes), "meal": refeicao,
        "country": pais or None, "market_segment": segmento, "distribution_channel": canal,
        "is_repeated_guest": int(repetido), "previous_cancellations": int(canc_prev),
        "previous_bookings_not_canceled": int(nao_canc), "reserved_room_type": quarto,
        "deposit_type": deposito, "agent": int(agente) if agente else None, "company": None,
        "days_in_waiting_list": int(espera), "customer_type": cliente, "adr": float(adr),
        "required_car_parking_spaces": int(vagas), "total_of_special_requests": int(pedidos),
    }
    return reserva if enviar or "reserva_atual" in st.session_state else None, enviar


def pagina_simular() -> None:
    st.title("Simular uma reserva nova")
    st.markdown(
        "Preencha o que se sabe **no ato da reserva** (sem o desfecho e sem colunas que só existem depois). "
        "A reserva passa pelo mesmo ETL do treino, no modo inferência, e pelo modelo vigente "
        "(`src.prever.prever_reserva`)."
    )
    exigir_modelo()
    reserva, enviar = formulario_reserva()
    if enviar:
        st.session_state["reserva_atual"] = reserva
    reserva = st.session_state.get("reserva_atual")
    if not reserva:
        return

    try:
        res = prever_reserva(reserva, modelo, vigente, cfg(), RAIZ).iloc[0]
    except ValueError as e:
        st.error(f"Entrada recusada pelo pipeline: {e}")
        return

    prob = float(res["prob_cancelamento"])
    alto = prob >= promocao.LIMIAR_CLASSE
    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Chance de cancelamento", pct(prob))
        (st.error if alto else st.success)(
            f"**{res['risco'].capitalize()}** (limiar de {pct(promocao.LIMIAR_CLASSE, 0)}, o mesmo da avaliação)")
        if res.get("aviso"):
            st.warning(f"Aviso: {res['aviso']}. A probabilidade pode não ser confiável.")
        if alto:
            st.markdown("**Ação sugerida:** pedir depósito, confirmar por contato ou oferecer tarifa não reembolsável.")
        else:
            st.markdown("**Ação sugerida:** nenhuma; conta como ocupação provável no planejamento.")
    with r2:
        barra = pd.DataFrame({"x": [prob], "y": [""]})
        fundo = alt.Chart(pd.DataFrame({"x": [1.0], "y": [""]})).mark_bar(color="#e0e0e0", size=36).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)), y=alt.Y("y:N", title=None))
        frente = alt.Chart(barra).mark_bar(color="#e53935" if alto else "#43a047", size=36).encode(x="x:Q", y="y:N")
        corte = alt.Chart(pd.DataFrame({"x": [promocao.LIMIAR_CLASSE]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="x:Q")
        st.altair_chart((fundo + frente + corte).properties(height=90), width="stretch")

    st.subheader("E se…? Sensibilidade a um atributo")
    st.caption("Mantém o resto da reserva e varia só um atributo, para mostrar o que o modelo aprendeu.")
    atributo = st.radio("Atributo", ["Antecedência (dias)", "Diária média (adr)", "Tipo de depósito", "Pedidos especiais"],
                        horizontal=True)
    if atributo == "Tipo de depósito":
        valores, campo = cfg()["inferencia"]["dominio_categorias"]["deposit_type"], "deposit_type"
    elif atributo == "Pedidos especiais":
        valores, campo = list(range(0, 6)), "total_of_special_requests"
    elif atributo == "Diária média (adr)":
        valores, campo = list(range(0, 401, 10)), "adr"
    else:
        valores, campo = list(range(0, 366, 5)), "lead_time"
    variacoes = [{**reserva, campo: v} for v in valores]
    curva = prever_reserva(variacoes, modelo, vigente, cfg(), RAIZ)
    curva["valor"] = valores
    if isinstance(valores[0], str):
        grafico = alt.Chart(curva).mark_bar().encode(
            x=alt.X("valor:N", title=atributo), y=alt.Y("prob_cancelamento:Q", title="Chance de cancelamento",
                                                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.prob_cancelamento >= 0.5, alt.value("#e53935"), alt.value("#43a047")),
            tooltip=[alt.Tooltip("valor:N", title=atributo), alt.Tooltip("prob_cancelamento:Q", format=".1%", title="chance")])
    else:
        grafico = alt.Chart(curva).mark_line(point=True).encode(
            x=alt.X("valor:Q", title=atributo), y=alt.Y("prob_cancelamento:Q", title="Chance de cancelamento",
                                                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("valor:Q", title=atributo), alt.Tooltip("prob_cancelamento:Q", format=".1%", title="chance")])
    limiar = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((grafico + limiar).properties(height=320), width="stretch")

    with st.expander("Ver o JSON enviado ao modelo"):
        st.json(reserva)


def pagina_lote() -> None:
    st.title("Previsões × realidade")
    st.markdown(
        "Em cada ciclo do GitHub Actions, o modelo em uso previu o **lote do mês seguinte**, que ainda não tinha "
        "entrado no pipeline (notebook 05). Como o dataset é histórico, dá para conferir essas previsões com o que "
        "aconteceu de verdade. É a avaliação mais honesta do modelo: dados que não existiam quando ele foi treinado."
    )
    arquivos = previsoes_actions()
    if not arquivos:
        st.warning("Nenhuma previsão do Actions em `evidencias/actions/previsoes/`.")
        return
    c = cfg()

    resumo = []
    for lote, arq in arquivos.items():
        p = conferir_previsoes(str(arq))
        m = metricas_classificacao(p["is_canceled"], p["previsao"], p["prob_cancelamento"])
        resumo.append({"lote": lote.removeprefix("lote_"), "F1 macro": m["f1_macro"], "ROC-AUC": m["roc_auc"],
                       "modelo": f"B · {dv_curto(p['data_version'].iloc[0])}"})
    resumo = pd.DataFrame(resumo)
    st.subheader("F1 macro de cada previsão, mês a mês")
    base = alt.Chart(resumo).encode(x=alt.X("lote:N", title="Mês previsto", axis=alt.Axis(labelAngle=0)))
    st.altair_chart(
        (base.mark_line(point=alt.OverlayMarkDef(size=80)).encode(
            y=alt.Y("F1 macro:Q", scale=alt.Scale(domain=[0.6, 0.8])),
            tooltip=["lote", "modelo", alt.Tooltip("F1 macro:Q", format=".3f"), alt.Tooltip("ROC-AUC:Q", format=".3f")])
         + base.mark_text(dy=-14).encode(y="F1 macro:Q", text=alt.Text("F1 macro:Q", format=".3f")))
        .properties(height=260), width="stretch")
    st.caption("Cada ponto é a previsão feita pelo modelo em uso no ciclo anterior. A queda em jul-ago/2017 "
               "acompanha a mudança no perfil das reservas apontada pelo monitoramento de drift.")

    st.divider()
    lotes = list(arquivos)
    escolhido = st.selectbox("Detalhar o mês", lotes, index=len(lotes) - 1,
                             format_func=lambda l: f"{l.removeprefix('lote_')}  ·  previsto pelo {resumo.loc[lotes.index(l), 'modelo']}")
    prev = conferir_previsoes(str(arquivos[escolhido]))
    y, prob = prev["is_canceled"].to_numpy(), prev["prob_cancelamento"].to_numpy()

    with st.container(border=True):
        st.markdown("**E se o corte fosse outro?** O pipeline usa 50%: a partir dele a reserva é \"risco alto\". "
                    "Baixar o corte pega mais cancelamentos, ao preço de mais alarmes falsos. É só uma simulação: "
                    "o modelo e a regra de promoção não mudam.")
        limiar = st.slider("Corte de decisão", 0.20, 0.80, float(promocao.LIMIAR_CLASSE), 0.05, format="%.2f")
        cortes = [round(0.20 + 0.05 * i, 2) for i in range(13)]
        curva = []
        for c_ in cortes:
            rc = resumo_confusao(y, (prob >= c_).astype(int))
            curva += [{"corte": c_, "taxa": rc["taxa_falso_negativo"], "erro": "cancelamentos perdidos"},
                      {"corte": c_, "taxa": rc["taxa_falso_positivo"], "erro": "alarmes falsos"}]
        linhas = alt.Chart(pd.DataFrame(curva)).mark_line(point=True).encode(
            x=alt.X("corte:Q", title="Corte de decisão", axis=alt.Axis(format="%")),
            y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Taxa de erro"),
            color=alt.Color("erro:N", scale=alt.Scale(range=["#fb8c00", "#e53935"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("corte:Q", format=".0%"), "erro", alt.Tooltip("taxa:Q", format=".1%")])
        marca = alt.Chart(pd.DataFrame({"x": [limiar]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((linhas + marca).properties(height=220), width="stretch")
        st.caption("Cancelamentos perdidos: % das reservas canceladas que o modelo deixou passar. "
                   "Alarmes falsos: % das reservas que vieram e foram marcadas como risco alto.")

    p = (prob >= limiar).astype(int)
    m = metricas_classificacao(y, p, prob)
    r = resumo_confusao(y, p)

    k = st.columns(5)
    k[0].metric("Reservas", f"{len(prev):,}".replace(",", "."))
    k[1].metric("F1 macro", num(m["f1_macro"]))
    k[2].metric("ROC-AUC", num(m["roc_auc"]))
    k[3].metric("Cancelamentos reais", pct(y.mean()))
    k[4].metric("Cancelamentos previstos", pct(prev["prob_cancelamento"].mean()),
                help="Média das probabilidades: a estimativa de quartos que vão ficar livres.")

    e1, e2 = st.columns([1, 1])
    with e1:
        st.subheader("Matriz de confusão")
        mat = matriz_confusao(y, p).reset_index().melt(id_vars="real", var_name="previsto", value_name="n")
        base = alt.Chart(mat).encode(x=alt.X("previsto:N", title="Previsto", axis=alt.Axis(labelAngle=0)),
                                     y=alt.Y("real:N", title="Real"))
        st.altair_chart(
            (base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
             + base.mark_text(fontSize=22).encode(text="n:Q", color=alt.condition(alt.datum.n > mat["n"].max() / 2,
                                                                               alt.value("white"), alt.value("black"))))
            .properties(height=280), width="stretch")
        st.caption(
            f"Erro total {pct(r['taxa_erro'])}. Cancelamentos não previstos: {pct(r['taxa_falso_negativo'])} "
            f"dos cancelados. Alarme falso: {pct(r['taxa_falso_positivo'])} de quem veio.")
    with e2:
        st.subheader("Previsto × real por semana")
        sem = prev.assign(semana=pd.to_datetime(prev["arrival_date"]).dt.to_period("W").dt.start_time)
        sem = sem.groupby("semana").agg(real=("is_canceled", "mean"), previsto=("prob_cancelamento", "mean")).reset_index()
        sem = sem.melt("semana", var_name="série", value_name="taxa")
        st.altair_chart(alt.Chart(sem).mark_line(point=True).encode(
            x=alt.X("semana:T", title="Semana de chegada"),
            y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Taxa de cancelamento"),
            color=alt.Color("série:N", title=None)).properties(height=300), width="stretch")
        st.caption("'previsto' é a média das probabilidades: é a estimativa de quartos que vão ficar livres.")

    st.subheader("Onde o modelo erra")
    grupo = st.selectbox("Agrupar por", ["deposit_type", "market_segment", "hotel", "customer_type",
                                          "distribution_channel", "faixa de antecedência"])
    serie = (faixa_lead_time(prev["lead_time"], c["analise_erros"]["faixas_lead_time"]).rename("faixa_lead_time")
             if grupo.startswith("faixa") else prev[grupo])
    tab = erros_por_grupo(serie, y, p, r["falsos_positivos"] + r["falsos_negativos"], 50)
    st.dataframe(
        tab.drop(columns="grupo"), hide_index=True, width="stretch",
        column_config={
            "taxa_cancelamento_real": st.column_config.NumberColumn("cancel. real", format="percent"),
            "taxa_cancelamento_prevista": st.column_config.NumberColumn("cancel. previsto", format="percent"),
            "taxa_erro": st.column_config.ProgressColumn("taxa de erro", format="percent", min_value=0, max_value=1),
            "parcela_dos_erros": st.column_config.NumberColumn("parcela dos erros", format="percent"),
        })

    with st.expander("Reservas do lote com a probabilidade prevista"):
        st.dataframe(prev.sort_values("prob_cancelamento", ascending=False), hide_index=True, width="stretch")


@st.cache_data(show_spinner="Lendo os lotes brutos…")
def dados_brutos() -> pd.DataFrame:
    c = cfg()
    arquivos = sorted((RAIZ / c["dados"]["dir_raw"]).glob("lote_*.csv.gz"))
    arquivos += sorted((RAIZ / c["dados"]["dir_incoming"]).glob("lote_*.csv.gz"))
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)


def grafico_taxa(df: pd.DataFrame, coluna: str, titulo: str, ordem: list | None = None, min_n: int = 200):
    t = df.groupby(coluna, observed=True)["is_canceled"].agg(taxa="mean", n="size").reset_index()
    t = t[t["n"] >= min_n]
    t[coluna] = t[coluna].astype(str)
    media = df["is_canceled"].mean()
    barras = alt.Chart(t).mark_bar().encode(
        y=alt.Y(f"{coluna}:N", sort=ordem or "-x", title=None),
        x=alt.X("taxa:Q", axis=alt.Axis(format="%"), title="Taxa de cancelamento", scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(alt.datum.taxa >= media, alt.value("#e53935"), alt.value("#5c6bc0")),
        tooltip=[alt.Tooltip(f"{coluna}:N", title=titulo), alt.Tooltip("taxa:Q", format=".1%"), alt.Tooltip("n:Q", title="reservas")])
    rotulos = barras.mark_text(align="left", dx=4).encode(text=alt.Text("taxa:Q", format=".0%"), color=alt.value("gray"))
    linha = alt.Chart(pd.DataFrame({"x": [media]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    return (barras + rotulos + linha).properties(height=max(140, 30 * len(t)), title=titulo)


def pagina_exploracao() -> None:
    st.title("Exploração dos dados")
    st.markdown(
        "*Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019): reservas reais de dois hotéis em Portugal, "
        "com chegadas de jul/2015 a ago/2017, divididas em 26 lotes mensais. A linha tracejada nos gráficos é a "
        "taxa média de cancelamento; barras vermelhas ficam acima dela."
    )
    bruto = dados_brutos()
    df = todos_os_lotes()

    k = st.columns(4)
    k[0].metric("Reservas (brutas)", f"{len(bruto):,}".replace(",", "."))
    k[1].metric("Após o ETL", f"{len(df):,}".replace(",", "."), help="Remove reservas sem hóspedes e duplicatas reentregues entre lotes.")
    k[2].metric("Taxa de cancelamento", pct(bruto["is_canceled"].mean()))
    k[3].metric("Colunas originais", bruto.shape[1])

    st.subheader("Classes desbalanceadas, e por isso o F1 macro")
    c1, c2 = st.columns([1, 2])
    with c1:
        cls = bruto["is_canceled"].map({0: "não cancelou", 1: "cancelou"}).value_counts().reset_index()
        cls.columns = ["classe", "n"]
        st.altair_chart(alt.Chart(cls).mark_arc(innerRadius=55).encode(
            theta="n:Q", color=alt.Color("classe:N", scale=alt.Scale(range=["#e53935", "#5c6bc0"]),
                                          legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["classe", "n"]).properties(height=240), width="stretch")
    with c2:
        st.markdown(
            "- 63% das reservas vieram e 37% cancelaram. Um modelo que dissesse sempre \"vai vir\" teria 63% de "
            "acurácia sem servir para nada.\n"
            "- O **F1 macro** dá o mesmo peso às duas classes, então só sobe se o modelo acertar também os cancelamentos.\n"
            "- Os dois algoritmos usam `class_weight` balanceado pelo mesmo motivo.")

    st.subheader("Cancelamento ao longo do tempo")
    tempo = df.assign(mes=pd.to_datetime(df["arrival_date"]).dt.to_period("M").dt.start_time)
    tempo = tempo.groupby(["mes", "hotel"])["is_canceled"].mean().reset_index()
    st.altair_chart(alt.Chart(tempo).mark_line(point=True).encode(
        x=alt.X("mes:T", title="Mês de chegada"), y=alt.Y("is_canceled:Q", axis=alt.Axis(format="%"), title="Taxa de cancelamento"),
        color=alt.Color("hotel:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("mes:T", format="%m/%Y"), "hotel", alt.Tooltip("is_canceled:Q", format=".1%")],
    ).properties(height=300), width="stretch")
    st.caption("O City Hotel cancela bem mais que o Resort, e a taxa sobe em 2017. Por isso a partição é "
               "cronológica (treina no passado, testa no mais recente) e o drift compara com o mesmo mês do ano anterior.")

    st.subheader("O que separa quem cancela de quem vem")
    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(grafico_taxa(df, "deposit_type", "Tipo de depósito"), width="stretch")
        st.caption("Contraintuitivo: *Non Refund* cancela quase sempre. É o atributo mais votado na seleção.")
        st.altair_chart(grafico_taxa(df, "market_segment", "Segmento de mercado"), width="stretch")
    with g2:
        faixas = faixa_lead_time(df["lead_time"], cfg()["analise_erros"]["faixas_lead_time"])
        ordem = ["0-6", "7-29", "30-89", "90-179", "180-364", "365+"]
        st.altair_chart(grafico_taxa(df.assign(antecedencia=faixas), "antecedencia",
                                     "Antecedência (dias)", ordem=ordem), width="stretch")
        st.caption("Quanto mais cedo a reserva é feita, mais chance de ser cancelada.")
        pedidos = df.assign(pedidos=df["total_of_special_requests"].clip(upper=3).astype(int).astype(str).replace("3", "3+"))
        st.altair_chart(grafico_taxa(pedidos, "pedidos", "Pedidos especiais", ordem=["0", "1", "2", "3+"]), width="stretch")
        st.caption("Quem faz pedidos especiais está comprometido com a viagem e cancela menos.")

    st.subheader("Colunas que vazariam a resposta")
    st.markdown(
        "Algumas colunas só são preenchidas **depois** da reserva. Se entrassem no modelo, ele \"colaria\" a "
        "resposta e não funcionaria no momento em que a reserva chega. Por isso o ETL as remove.")
    vaz = pd.crosstab(bruto["reservation_status"], bruto["is_canceled"].map({0: "não cancelou", 1: "cancelou"}))
    v1, v2 = st.columns([1, 1])
    v1.dataframe(vaz, width="stretch")
    v2.markdown(
        "- `reservation_status`: *Canceled* e *No-Show* **são** o cancelamento.\n"
        "- `reservation_status_date`: a data do desfecho.\n"
        "- `assigned_room_type` e `booking_changes`: só existem no check-in ou depois de alterações.\n"
        "- `country`: o artigo do dataset avisa que costuma ser corrigido no check-in.")

    st.subheader("Valores ausentes e duplicatas")
    falt = bruto.isna().mean().loc[lambda s: s > 0].sort_values(ascending=False).reset_index()
    falt.columns = ["coluna", "ausentes"]
    d1, d2 = st.columns(2)
    d1.dataframe(falt, hide_index=True, width="stretch",
                 column_config={"ausentes": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    d2.markdown(
        "- `company` e `agent` ausentes significam \"sem empresa\" e \"sem agência\": viram os atributos "
        "`tem_empresa` e `tem_agente`, não são descartados.\n"
        f"- {pct(bruto.duplicated().mean(), 0)} das linhas são idênticas a outra. A maioria é de reservas de grupo "
        "canceladas juntas (comportamento real), então só sai a linha reentregue em outro lote.")


def pagina_ciclos() -> None:
    st.title("Ciclos do pipeline e promoção controlada")
    regra = cfg()["promocao"]
    st.markdown(
        f"**Regra fixada antes do treino:** o vencedor (A ou B, maior F1 macro na CV cronológica) só substitui o "
        f"vigente se tiver F1 macro ≥ {num(regra['piso_metrica'], 2)} no teste **e** superar o vigente em pelo "
        f"menos {num(regra['min_melhoria_abs'])}, avaliados nos mesmos registros."
    )
    runs = pd.read_csv(EVID / "mlflow" / "historico_runs.csv")
    runs["dv"] = runs["data_version"].map(dv_curto)

    hist = pd.DataFrame([{
        "Ciclo": h["cycle_id"], "Dados": dv_curto(h["data_version"]), "Vencedor": h["candidato"]["modelo"],
        "F1 vencedor (teste)": h.get("valor_vencedor"), "F1 vigente (teste)": h.get("valor_vigente"),
        "Ganho": (h["valor_vencedor"] - h["valor_vigente"]) if h.get("valor_vigente") is not None else None,
        "Decisão": h["decisao"], "Motivo": h["motivo"],
    } for h in manifesto.get("historico", [])])
    st.dataframe(hist, hide_index=True, width="stretch", column_config={
        "F1 vencedor (teste)": st.column_config.NumberColumn(format="%.4f"),
        "F1 vigente (teste)": st.column_config.NumberColumn(format="%.4f"),
        "Ganho": st.column_config.NumberColumn(format="%+.4f"),
    })

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Competição: F1 macro na CV")
        cv = runs[runs["etapa"].isin(["03_modelo_A", "03_modelo_B"])].copy()
        cv["modelo"] = cv["etapa"].str[-1].map(NOMES_MODELO)
        st.altair_chart(alt.Chart(cv).mark_bar().encode(
            x=alt.X("dv:N", title="Ciclo (data_version)"), xOffset="modelo:N",
            y=alt.Y("metrics.cv_f1_macro:Q", title="F1 macro (CV)", scale=alt.Scale(domain=[0.6, 0.85])),
            color=alt.Color("modelo:N", scale=alt.Scale(range=[COR_A, COR_B]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["modelo", alt.Tooltip("metrics.cv_f1_macro:Q", format=".4f"),
                     alt.Tooltip("metrics.cv_roc_auc:Q", format=".4f", title="ROC-AUC")],
        ).properties(height=320), width="stretch")
    with c2:
        st.subheader("Promoção: vencedor × vigente no teste")
        pr = hist.melt(id_vars=["Dados", "Decisão"], value_vars=["F1 vencedor (teste)", "F1 vigente (teste)"],
                       var_name="série", value_name="F1").dropna()
        st.altair_chart(alt.Chart(pr).mark_line(point=alt.OverlayMarkDef(size=90)).encode(
            x=alt.X("Dados:N", title="Ciclo (data_version)"),
            y=alt.Y("F1:Q", scale=alt.Scale(domain=[0.70, 0.77]), title="F1 macro (teste do ciclo)"),
            color=alt.Color("série:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Dados", "série", alt.Tooltip("F1:Q", format=".4f"), "Decisão"],
        ).properties(height=320), width="stretch")

    st.subheader("Todas as métricas do MLflow")
    metrica = st.selectbox("Métrica da CV", [c.removeprefix("metrics.cv_") for c in runs.columns if c.startswith("metrics.cv_")])
    cv = runs[runs["etapa"].isin(["03_modelo_A", "03_modelo_B"])].copy()
    tabela = cv.pivot_table(index="dv", columns="etapa", values=f"metrics.cv_{metrica}").rename(
        columns={"03_modelo_A": NOMES_MODELO["A"], "03_modelo_B": NOMES_MODELO["B"]})
    st.dataframe(tabela.style.format("{:.4f}").highlight_max(axis=1, color="#ffe0b2"), width="stretch")
    with st.expander("Runs do MLflow (cópia de `evidencias/actions/mlflow/historico_runs.csv`)"):
        st.dataframe(runs.drop(columns="dv"), hide_index=True, width="stretch")
        st.caption("A execução sem runs de modelo (dv020, só `02_votacao`) é a falha simulada: o fluxo parou antes da promoção.")


def pagina_features() -> None:
    st.title("Seleção de features por votação")
    v = cfg()["selecao"]
    st.markdown(
        f"Três métodos ranqueiam as features candidatas e cada um vota nas suas **{v['k_top']} melhores**: "
        "**SelectKBest** (informação mútua), **Random Forest** (importância) e **RFE** com regressão logística. "
        f"É eleita a feature com pelo menos **{v['votacao']['min_votos']} dos 3 votos**; se ficarem menos de "
        f"{v['votacao']['min_features']}, completa pelo ranking médio."
    )
    arquivos = sorted((EVID / "votacao").glob("votos_*.csv"))
    rotulos = {}
    for a in arquivos:
        dv = pd.read_csv(a, usecols=["data_version"]).iloc[0, 0]
        rotulos[f"{dv_curto(dv)} · {a.stem.removeprefix('votos_')}"] = a
    escolha = st.selectbox("Ciclo", list(rotulos), index=len(rotulos) - 1)
    votos = pd.read_csv(rotulos[escolha])

    k = st.columns(4)
    k[0].metric("Candidatas (após one-hot)", len(votos))
    k[1].metric("Eleitas", int(votos["eleita"].sum()))
    k[2].metric("Unânimes (3 votos)", int((votos["votos"] == 3).sum()))
    k[3].metric("Sem voto", int((votos["votos"] == 0).sum()))

    top = votos.sort_values("ranking_medio").head(35)
    longo = top.melt(id_vars=["feature", "ranking_medio", "eleita"],
                     value_vars=["voto_selectkbest", "voto_random_forest", "voto_rfe"], var_name="método", value_name="voto")
    longo["método"] = longo["método"].str.removeprefix("voto_")
    longo = longo[longo["voto"] == 1]
    ordem = top["feature"].tolist()
    st.altair_chart(alt.Chart(longo).mark_bar().encode(
        y=alt.Y("feature:N", sort=ordem, title=None),
        x=alt.X("count():Q", title="Votos", scale=alt.Scale(domain=[0, 3]), axis=alt.Axis(tickCount=3)),
        color=alt.Color("método:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["feature", "método"],
    ).properties(height=max(300, 18 * len(ordem))), width="stretch")
    st.caption("As 35 primeiras pelo ranking médio. A linha de corte da eleição é 2 votos.")

    with st.expander("Tabela completa de votos"):
        st.dataframe(votos.drop(columns=["cycle_id", "data_version"]), hide_index=True, width="stretch")

    if vigente.get("features"):
        st.subheader("Features do modelo vigente")
        st.write(" · ".join(f"`{f}`" for f in vigente["features"]))


def pagina_drift() -> None:
    c = cfg()["drift"]
    st.title("Monitoramento de drift")
    st.markdown(
        "Cada lote novo é comparado com o **mesmo mês do ano anterior** (reservas são sazonais). Para cada feature "
        f"calcula-se o **PSI**: < {num(c['psi']['atencao'], 2)} estável, até {num(c['psi']['alerta'], 2)} atenção, "
        f"acima disso alerta. A taxa de cancelamento alerta a partir de {num(c['alvo']['alerta_pp'], 1)} p.p. de variação. "
        f"`{', '.join(c['informativas'])}` é só informativa (não entra no status)."
    )
    df = todos_os_lotes()
    lotes = sorted(df["lote_id"].unique())
    atual = st.select_slider("Lote analisado", options=lotes, value="lote_2017-04" if "lote_2017-04" in lotes else lotes[-1])
    tabela, resumo = monitorar_lote(df, atual, cfg(), lotes)

    status_cor = {"ok": "🟢", "atencao": "🟡", "alerta": "🔴", "sem_referencia": "⚪", "amostra_insuficiente": "⚪"}
    k = st.columns(4)
    k[0].metric("Status geral", f"{status_cor.get(resumo['status'], '')} {resumo['status']}")
    k[1].metric("Referência", ", ".join(resumo["referencia"]["lotes"][-1:]) or "—",
                help=f"tipo: {resumo['referencia']['tipo']}")
    if resumo.get("alvo"):
        a = resumo["alvo"]
        k[2].metric("Taxa de cancelamento", pct(a["taxa_atual"]), f"{a['diferenca_pp']:+.1f} p.p.".replace(".", ","),
                    delta_color="inverse")
        k[3].metric("Features em alerta", resumo["n_features_alerta"])
    if resumo["status"] == "sem_referencia":
        st.info("Primeiro lote: ainda não há referência para comparar.")
        return

    st.subheader(f"PSI por feature · {atual} contra {', '.join(resumo['referencia']['lotes'])}")
    t = tabela.copy()
    st.altair_chart(alt.Chart(t).mark_bar().encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("psi:Q", title="PSI"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["ok", "atencao", "alerta"],
                                                    range=["#43a047", "#fbc02d", "#e53935"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["feature", "tipo", alt.Tooltip("psi:Q", format=".3f"), "status"],
    ).properties(height=max(280, 18 * len(t))) + alt.Chart(pd.DataFrame({"x": [c["psi"]["alerta"]]})).mark_rule(
        strokeDash=[4, 4]).encode(x="x:Q"), width="stretch")

    st.subheader("Histórico (evidência do GitHub Actions)")
    hist = pd.read_csv(sorted((EVID / "drift").glob("historico_*.csv"))[-1]).dropna(subset=["taxa_atual"])
    longo = hist.melt(id_vars=["lote", "status_alvo"], value_vars=["taxa_referencia", "taxa_atual"],
                      var_name="série", value_name="taxa")
    longo["série"] = longo["série"].map({"taxa_referencia": "referência", "taxa_atual": "lote"})
    st.altair_chart(alt.Chart(longo).mark_line(point=True).encode(
        x=alt.X("lote:N", title=None, axis=alt.Axis(labelAngle=-60)),
        y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Taxa de cancelamento", scale=alt.Scale(zero=False)),
        color=alt.Color("série:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["lote", "série", alt.Tooltip("taxa:Q", format=".1%"), "status_alvo"],
    ).properties(height=320), width="stretch")
    st.caption("Histórico gravado pela última execução do Actions, já com a regra atual (`agent` só informativo). "
               "O gráfico de PSI acima é calculado na hora, a partir dos lotes.")


{
    "Visão geral": pagina_visao_geral,
    "Simular uma reserva": pagina_simular,
    "Exploração dos dados": pagina_exploracao,
    "Previsões × realidade": pagina_lote,
    "Ciclos e promoção": pagina_ciclos,
    "Seleção de features": pagina_features,
    "Monitoramento de drift": pagina_drift,
}[pagina]()
