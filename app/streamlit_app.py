"""Interface de demonstração do pipeline MLOps (previsão de cancelamento de reservas).

Uso: ``streamlit run app/streamlit_app.py`` a partir da raiz do projeto.

Feita para apresentar: cada página tem poucos números, um gráfico e uma frase de conclusão, e o que é técnico
fica em "Detalhes técnicos". Só lê: o manifesto do vigente do GitHub Actions, as evidências copiadas dele e os
lotes de ``data/``. Não move lotes, não treina e não grava nada no repositório (a única escrita possível é
baixar o arquivo do vigente da Release ``modelos`` para ``models/registro/``, que o git ignora). As previsões
usam as mesmas funções do pipeline (``src.prever`` e ``src.promocao``).
"""
from __future__ import annotations

import json
import sys
import urllib.request
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

MANIFESTO = RAIZ / "models" / "modelo_vigente_actions.json"
# A Release `modelos` é pública: o arquivo do vigente é baixado daqui quando não está em models/registro/
# (num clone novo ou no Streamlit Community Cloud, já que models/registro/ não é versionado).
URL_RELEASE = "https://github.com/GabrielKaring/projeto-n3-mlops-hotel/releases/download/modelos/"
EVID = RAIZ / "evidencias" / "actions"
MESES_NOME = {v: k for k, v in MESES.items()}
MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
VERMELHO, AZUL, VERDE, LARANJA = "#c62839", "#5c6bc0", "#2e7d32", "#ef6c00"

st.set_page_config(page_title="N3 · Cancelamento de reservas", layout="wide")
st.markdown("""<style>
[data-testid="stMetricValue"] { font-size: 2.6rem; }
.frase { font-size: 1.2rem; font-weight: 600; padding: .6rem 0 .2rem 0; }
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- utilitários


def pct(valor: float, casas: int = 0) -> str:
    return f"{valor:.{casas}%}".replace(".", ",")


def num(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def mes_pt(lote: str) -> str:
    """``lote_2017-08`` -> ``ago/2017``."""
    ano, mes = lote.removeprefix("lote_").split("-")
    return f"{MESES_PT[int(mes) - 1]}/{ano}"


def frase(texto: str) -> None:
    """A conclusão de cada gráfico, em destaque."""
    st.markdown(f'<div class="frase">{texto}</div>', unsafe_allow_html=True)


@st.cache_data
def cfg() -> dict:
    return carregar_config()


@st.cache_data
def ler_manifesto() -> dict:
    return json.loads(MANIFESTO.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Carregando o modelo (na primeira vez, baixa uns 20 MB da Release do GitHub)…")
def carregar_modelo():
    manifesto = ler_manifesto()
    arquivo = (manifesto.get("vigente") or {}).get("arquivo")
    destino = RAIZ / promocao.DIR_REGISTRO / arquivo if arquivo else None
    if destino is not None and not destino.is_file():
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_suffix(".parcial")
        urllib.request.urlretrieve(URL_RELEASE + arquivo, temporario)
        temporario.replace(destino)
    return promocao.carregar_vigente(manifesto, RAIZ)


@st.cache_data(show_spinner="Lendo as reservas…")
def dados_brutos() -> pd.DataFrame:
    c = cfg()
    arquivos = sorted((RAIZ / c["dados"]["dir_raw"]).glob("lote_*.csv.gz"))
    arquivos += sorted((RAIZ / c["dados"]["dir_incoming"]).glob("lote_*.csv.gz"))
    partes = []
    for arq in arquivos:
        bruto = pd.read_csv(arq)
        bruto.insert(0, "linha_origem", range(len(bruto)))
        bruto.insert(0, "lote_id", arq.name.removesuffix(".csv.gz"))
        partes.append(bruto)
    return pd.concat(partes, ignore_index=True)


@st.cache_data(show_spinner="Preparando as reservas…")
def todos_os_lotes() -> pd.DataFrame:
    """Todos os lotes mensais com o mesmo ETL do pipeline, com rótulo."""
    limpo, _ = limpar_e_engenhar(dados_brutos(), cfg())
    return limpo


COLS_CONFERENCIA = ["is_canceled", "hotel", "market_segment", "deposit_type", "customer_type",
                    "distribution_channel", "lead_time"]


@st.cache_data
def previsoes_actions() -> dict[str, Path]:
    """Lote previsto -> arquivo de previsões do ciclo que o previu (cada ciclo prevê o mês seguinte)."""
    saida = {}
    for arq in sorted((EVID / "previsoes").glob("previsoes_*.csv.gz")):
        saida[pd.read_csv(arq, usecols=["lote_id"], nrows=1).iloc[0, 0]] = arq
    return saida


@st.cache_data(show_spinner="Conferindo as previsões com o que aconteceu…")
def conferir_previsoes(arquivo: str) -> pd.DataFrame:
    """Previsões gravadas no Actions + o desfecho real do lote (o mesmo cálculo do notebook 05)."""
    prev = pd.read_csv(arquivo)
    c = cfg()
    lote = prev["lote_id"].iloc[0]
    origem = next(p for d in (c["dados"]["dir_incoming"], c["dados"]["dir_raw"])
                  if (p := RAIZ / d / f"{lote}.csv.gz").exists())
    df, _ = promocao.preparar_lote(origem, c)
    return prev.merge(df[["lote_id", "linha_origem", *COLS_CONFERENCIA]], on=["lote_id", "linha_origem"])


def grafico_barras(tabela: pd.DataFrame, x: str, titulo: str, media: float, ordem: list | None = None):
    """Taxa de cancelamento por categoria: vermelho acima da média, azul abaixo."""
    base = alt.Chart(tabela).encode(
        x=alt.X(f"{x}:N", sort=ordem, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12, labelOverlap=False),
                scale=alt.Scale(paddingInner=0.25)),
        y=alt.Y("taxa:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1.05])))
    barras = base.mark_bar().encode(
        color=alt.condition(alt.datum.taxa >= media, alt.value(VERMELHO), alt.value(AZUL)),
        tooltip=[alt.Tooltip(f"{x}:N", title=titulo), alt.Tooltip("taxa:Q", format=".0%", title="cancelam")])
    rotulos = base.mark_text(dy=-10, fontSize=15, fontWeight="bold").encode(text=alt.Text("taxa:Q", format=".0%"))
    linha = alt.Chart(pd.DataFrame({"y": [media]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")
    return (barras + rotulos + linha).properties(height=280, title=alt.TitleParams(titulo, fontSize=16))


def taxa_por(df: pd.DataFrame, coluna: pd.Series | str, min_n: int = 200) -> pd.DataFrame:
    chave = df[coluna] if isinstance(coluna, str) else coluna
    t = df.groupby(chave.rename("grupo"), observed=True)["is_canceled"].agg(taxa="mean", n="size").reset_index()
    t["grupo"] = t["grupo"].astype(str)
    return t[t["n"] >= min_n]


# --------------------------------------------------------------------------- barra lateral

manifesto = ler_manifesto()
vigente = manifesto.get("vigente") or {}
modelo = None
erro_modelo = ""
try:
    modelo, vigente = carregar_modelo()
except (FileNotFoundError, OSError) as erro:  # sem internet ou arquivo ausente na Release
    erro_modelo = str(erro)

st.sidebar.title("Projeto N3")
st.sidebar.caption("Quais reservas de hotel vão ser canceladas?")
PAGINAS = {
    "O projeto": "projeto",
    "Os dados": "dados",
    "Testar uma reserva": "testar",
    "O modelo acertou?": "acertou",
    "Como o modelo é escolhido": "escolha",
    "Alerta de mudança": "alerta",
}
pagina = PAGINAS[st.sidebar.radio("Navegação", list(PAGINAS), label_visibility="collapsed")]

with st.sidebar.container(border=True):
    promovido = (vigente.get("promovido_em") or "")[:10]
    data_br = "/".join(reversed(promovido.split("-"))) if promovido else "?"
    st.markdown("**Modelo em uso**")
    st.caption(f"Random Forest, escolhido automaticamente em {data_br}, com dados até ago/2017.")
    if modelo is None:
        st.error("Não foi possível baixar o modelo.")
        if st.button("Tentar de novo", width="stretch"):
            carregar_modelo.clear()
            st.rerun()
st.sidebar.caption("github.com/GabrielKaring/projeto-n3-mlops-hotel")


def exigir_modelo() -> None:
    if modelo is None:
        st.warning(f"Esta página precisa do arquivo do modelo, que não pôde ser baixado ({erro_modelo}). "
                   "Confira a internet e use **Tentar de novo** na barra lateral, ou rode na raiz do projeto:")
        st.code(f"gh release download modelos --pattern {vigente.get('arquivo')} --dir models/registro")
        st.stop()


# --------------------------------------------------------------------------- páginas


def pagina_projeto() -> None:
    st.title("Quais reservas vão ser canceladas?")
    st.markdown("#### Se o hotel aceita reservas demais, falta quarto no balcão. Se aceita de menos, sobra quarto "
                "vazio. O modelo dá a **chance de cancelamento de cada reserva assim que ela entra**.")

    hist = manifesto.get("historico", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservas reais", "119 mil", help="2 hotéis em Portugal, de jul/2015 a ago/2017 (Kaggle)")
    c2.metric("Foram canceladas", pct(dados_brutos()["is_canceled"].mean()))
    c3.metric("Ciclos automáticos", len(hist), help="Execuções do pipeline no GitHub Actions que decidiram um modelo")
    c4.metric("Trocas de modelo", sum(h["decisao"] == "promovido" for h in hist),
              help="O modelo só é trocado quando o novo é melhor")

    st.subheader("Como funciona, todo dia, sem ninguém mexer")
    etapas = [("1", "Chegam dados novos", "Um mês de reservas por vez"),
              ("2", "Preparação", "Tira o que só se sabe depois da reserva"),
              ("3", "Dois modelos competem", "Regressão logística × Random Forest"),
              ("4", "Só troca se melhorar", "Regra definida antes de treinar"),
              ("5", "Previsões", "Para as reservas do mês seguinte")]
    cols = st.columns(len(etapas))
    for col, (passo, titulo, texto) in zip(cols, etapas):
        with col.container(border=True):
            st.markdown(f'<div style="font-size:1.8rem;font-weight:700;color:#8A1F3A">{passo}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**{titulo}**")
            st.caption(texto)

    st.subheader("Quem usa")
    q1, q2, q3 = st.columns(3)
    q1.info("**Gestão de receita**  \nQuanto overbooking aceitar em cada data")
    q2.info("**Reservas**  \nPedir depósito ou confirmar as reservas de risco alto")
    q3.info("**Operação**  \nPrever a ocupação real para escalar equipe e compras")

    with st.expander("Quem fez o quê"):
        import yaml

        dados = yaml.safe_load((RAIZ / "config" / "equipe.yaml").read_text(encoding="utf-8"))
        linhas = []
        for fid, info in (dados.get("features") or {}).items():
            donos = info.get("dono", "")
            linhas.append({"Parte": f"{fid} · {info.get('titulo', '')}",
                           "Quem": ", ".join(donos) if isinstance(donos, list) else donos})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")


def pagina_dados() -> None:
    st.title("O que os dados mostram")
    df = todos_os_lotes()
    media = df["is_canceled"].mean()
    st.markdown(f"Em média, **{pct(media)} das reservas são canceladas** (linha tracejada). "
                "Em vermelho, o que cancela mais que a média.")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.altair_chart(grafico_barras(taxa_por(df, "deposit_type"), "grupo", "Tipo de depósito", media), width="stretch")
        frase("Sem reembolso cancela quase sempre")
    with g2:
        ordem = ["0-6", "7-29", "30-89", "90-179", "180-364", "365+"]
        faixas = faixa_lead_time(df["lead_time"], cfg()["analise_erros"]["faixas_lead_time"])
        st.altair_chart(grafico_barras(taxa_por(df, faixas), "grupo", "Dias de antecedência", media, ordem), width="stretch")
        frase("Quanto mais cedo reserva, mais cancela")
    with g3:
        pedidos = df["total_of_special_requests"].clip(upper=3).astype(int).astype(str).replace("3", "3+")
        st.altair_chart(grafico_barras(taxa_por(df, pedidos), "grupo", "Pedidos especiais", media, ["0", "1", "2", "3+"]),
                        width="stretch")
        frase("Quem faz pedidos especiais vem")

    st.subheader("E muda com o tempo")
    tempo = df.assign(mes=pd.to_datetime(df["arrival_date"]).dt.to_period("M").dt.start_time)
    tempo = tempo.groupby(["mes", "hotel"])["is_canceled"].mean().reset_index()
    st.altair_chart(alt.Chart(tempo).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%m/%Y")), y=alt.Y("is_canceled:Q", axis=alt.Axis(format="%"), title="Cancelamento"),
        color=alt.Color("hotel:N", scale=alt.Scale(range=[VERMELHO, AZUL]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("mes:T", format="%m/%Y"), "hotel", alt.Tooltip("is_canceled:Q", format=".0%")],
    ).properties(height=280), width="stretch")
    frase("O hotel da cidade cancela mais, e o cancelamento sobe em 2017. Por isso o modelo é treinado no passado "
          "e testado nos meses mais recentes.")

    with st.expander("Detalhes técnicos"):
        bruto = dados_brutos()
        mil = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
        st.markdown(
            f"- **{mil(len(bruto))} reservas** brutas, {bruto.shape[1] - 2} colunas; {mil(len(df))} depois da preparação "
            "(saem reservas sem hóspedes e linhas reentregues em outro lote).\n"
            "- **Classes desbalanceadas** (63% × 37%): por isso a métrica é o F1 macro, que pesa as duas classes igual.\n"
            "- **Colunas que vazariam a resposta** saem antes do treino: `reservation_status` e "
            "`reservation_status_date` (o desfecho), `assigned_room_type`, `booking_changes` e `country` "
            "(só se sabem no check-in ou depois).\n"
            "- `company` (94%) e `agent` (14%) vazios significam \"sem empresa\" e \"sem agência\" e viram os atributos "
            "`tem_empresa` e `tem_agente`.")
        vaz = pd.crosstab(bruto["reservation_status"], bruto["is_canceled"].map({0: "veio", 1: "cancelou"}))
        st.caption("Por que `reservation_status` sai: ele já é a resposta.")
        st.dataframe(vaz, width="content")


EXEMPLOS = [("Reserva online, 120 dias antes", 0), ("Grupo sem reembolso", 1), ("Hóspede fiel, reserva direta", 2)]


def pagina_testar() -> None:
    st.title("Testar uma reserva nova")
    st.markdown("Escolha um exemplo ou mude os campos: a chance de cancelamento é recalculada na hora.")
    exigir_modelo()
    c = cfg()["inferencia"]["dominio_categorias"]
    exemplos = json.loads((RAIZ / "exemplos" / "reservas_exemplo.json").read_text(encoding="utf-8"))

    if "exemplo" not in st.session_state:
        st.session_state["exemplo"] = 0
    b = st.columns(len(EXEMPLOS))
    for col, (rotulo, i) in zip(b, EXEMPLOS):
        if col.button(rotulo, width="stretch", type="primary" if st.session_state["exemplo"] == i else "secondary"):
            st.session_state["exemplo"] = i
            st.rerun()
    base = exemplos[st.session_state["exemplo"]]
    k = f"e{st.session_state['exemplo']}"  # a chave muda com o exemplo, então os campos são recarregados

    def idx(campo: str) -> int:
        return c[campo].index(base[campo]) if base.get(campo) in c[campo] else 0

    esq, dir_ = st.columns([3, 2], gap="large")
    with esq:
        a1, a2, a3 = st.columns(3)
        hotel = a1.selectbox("Hotel", c["hotel"], index=idx("hotel"), key=f"hotel{k}")
        chegada = a2.date_input("Chegada", key=f"data{k}", value=date(
            int(base["arrival_date_year"]), MESES[base["arrival_date_month"]], int(base["arrival_date_day_of_month"])))
        lead = a3.number_input("Dias de antecedência", 0, 800, int(base["lead_time"]), key=f"lead{k}")
        b1, b2, b3 = st.columns(3)
        deposito = b1.selectbox("Depósito", c["deposit_type"], index=idx("deposit_type"), key=f"dep{k}")
        segmento = b2.selectbox("Como reservou", c["market_segment"], index=idx("market_segment"), key=f"seg{k}")
        pedidos = b3.number_input("Pedidos especiais", 0, 5, int(base["total_of_special_requests"]), key=f"ped{k}")
        d1, d2, d3 = st.columns(3)
        noites = d1.number_input("Noites", 1, 60, int(base["stays_in_weekend_nights"] + base["stays_in_week_nights"]),
                                 key=f"noi{k}")
        adultos = d2.number_input("Adultos", 1, 10, max(1, int(base["adults"])), key=f"ad{k}")
        adr = d3.number_input("Diária (€)", 0.0, 1000.0, float(base["adr"]), step=10.0, key=f"adr{k}")
        with st.expander("Mais campos"):
            e1, e2, e3 = st.columns(3)
            canal = e1.selectbox("Canal", c["distribution_channel"], index=idx("distribution_channel"), key=f"can{k}")
            cliente = e2.selectbox("Tipo de cliente", c["customer_type"], index=idx("customer_type"), key=f"cli{k}")
            refeicao = e3.selectbox("Refeição", c["meal"], index=idx("meal"), key=f"mea{k}")
            f1, f2, f3 = st.columns(3)
            quarto = f1.selectbox("Quarto", c["reserved_room_type"], index=idx("reserved_room_type"), key=f"qu{k}")
            criancas = f2.number_input("Crianças", 0, 10, int(base.get("children") or 0), key=f"cr{k}")
            vagas = f3.number_input("Vagas de garagem", 0, 3, int(base["required_car_parking_spaces"]), key=f"vag{k}")
            g1, g2, g3 = st.columns(3)
            repetido = g1.checkbox("Já se hospedou antes", bool(base["is_repeated_guest"]), key=f"rep{k}")
            canc_prev = g2.number_input("Cancelamentos anteriores", 0, 30, int(base["previous_cancellations"]), key=f"cp{k}")
            nao_canc = g3.number_input("Estadias anteriores", 0, 80, int(base["previous_bookings_not_canceled"]), key=f"nc{k}")

    # noites divididas como no exemplo: fim de semana primeiro, o resto durante a semana
    fds = min(int(base["stays_in_weekend_nights"]), int(noites))
    reserva = {
        "hotel": hotel, "lead_time": int(lead),
        "arrival_date_year": chegada.year, "arrival_date_month": MESES_NOME[chegada.month],
        "arrival_date_week_number": int(chegada.isocalendar()[1]), "arrival_date_day_of_month": chegada.day,
        "stays_in_weekend_nights": fds, "stays_in_week_nights": int(noites) - fds,
        "adults": int(adultos), "children": int(criancas), "babies": int(base["babies"]), "meal": refeicao,
        "country": base.get("country"), "market_segment": segmento, "distribution_channel": canal,
        "is_repeated_guest": int(repetido), "previous_cancellations": int(canc_prev),
        "previous_bookings_not_canceled": int(nao_canc), "reserved_room_type": quarto,
        "deposit_type": deposito, "agent": base.get("agent"), "company": None,
        "days_in_waiting_list": int(base["days_in_waiting_list"]), "customer_type": cliente, "adr": float(adr),
        "required_car_parking_spaces": int(vagas), "total_of_special_requests": int(pedidos),
    }
    try:
        res = prever_reserva(reserva, modelo, vigente, cfg(), RAIZ).iloc[0]
    except ValueError as e:
        dir_.error(f"Reserva recusada: {e}")
        return

    prob = float(res["prob_cancelamento"])
    alto = prob >= promocao.LIMIAR_CLASSE
    with dir_:
        with st.container(border=True):
            st.markdown("##### Chance de cancelamento")
            cor = VERMELHO if alto else VERDE
            st.markdown(f'<div style="font-size:4.5rem;font-weight:700;color:{cor};line-height:1.1">{pct(prob)}</div>',
                        unsafe_allow_html=True)
            st.progress(prob)
            if alto:
                st.error("**Risco alto.** Sugestão: pedir depósito ou confirmar por contato.")
            else:
                st.success("**Risco baixo.** Conta como ocupação provável.")
            if res.get("aviso"):
                st.warning(f"Aviso: {res['aviso']}.")
            st.caption("A partir de 50%, a reserva é considerada de risco alto.")

    st.subheader("E se mudasse a antecedência?")
    dias = list(range(0, 366, 5))
    curva = prever_reserva([{**reserva, "lead_time": d} for d in dias], modelo, vigente, cfg(), RAIZ)
    curva["dias"] = dias
    linha = alt.Chart(curva).mark_line(strokeWidth=3, color=VERMELHO).encode(
        x=alt.X("dias:Q", title="Dias de antecedência"),
        y=alt.Y("prob_cancelamento:Q", axis=alt.Axis(format="%"), title="Chance de cancelamento",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("dias:Q", title="dias"), alt.Tooltip("prob_cancelamento:Q", format=".0%", title="chance")])
    corte = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")
    atual = alt.Chart(pd.DataFrame({"dias": [int(lead)], "p": [prob]})).mark_point(size=180, filled=True, color="black").encode(
        x="dias:Q", y="p:Q")
    st.altair_chart((linha + corte + atual).properties(height=260), width="stretch")
    frase("O ponto preto é esta reserva. Mantendo o resto igual, só a antecedência muda a chance assim.")


def pagina_acertou() -> None:
    st.title("O modelo acertou?")
    st.markdown("Todo mês o modelo previu as reservas **do mês seguinte**, antes de elas acontecerem. "
                "Como os dados são históricos, dá para comparar com o que aconteceu de verdade.")
    arquivos = previsoes_actions()
    if not arquivos:
        st.warning("Nenhuma previsão do Actions em `evidencias/actions/previsoes/`.")
        return

    resumo = []
    for lote, arq in arquivos.items():
        p = conferir_previsoes(str(arq))
        r = resumo_confusao(p["is_canceled"], p["previsao"])
        m = metricas_classificacao(p["is_canceled"], p["previsao"])
        resumo.append({"mês": mes_pt(lote), "acertos": 1 - r["taxa_erro"], "f1": m["f1_macro"]})
    resumo = pd.DataFrame(resumo)
    base = alt.Chart(resumo).encode(x=alt.X("mês:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)))
    st.altair_chart(
        (base.mark_bar(size=50, color=AZUL).encode(y=alt.Y("acertos:Q", axis=alt.Axis(format="%"), title="Reservas acertadas",
                                                                 scale=alt.Scale(domain=[0, 1])),
                                                    tooltip=["mês", alt.Tooltip("acertos:Q", format=".0%")])
         + base.mark_text(dy=-10, fontSize=15, fontWeight="bold").encode(y="acertos:Q", text=alt.Text("acertos:Q", format=".0%")))
        .properties(height=260), width="stretch")
    frase(f"O modelo acertou entre {pct(resumo['acertos'].min())} e {pct(resumo['acertos'].max())} das reservas, "
          "em meses que ele nunca tinha visto.")

    st.divider()
    lotes = list(arquivos)
    escolhido = st.selectbox("Ver um mês em detalhe", lotes, index=len(lotes) - 1, format_func=mes_pt)
    prev = conferir_previsoes(str(arquivos[escolhido]))
    y, prob = prev["is_canceled"].to_numpy(), prev["prob_cancelamento"].to_numpy()

    with st.expander("E se o corte fosse outro? (hoje é 50%)"):
        st.caption("Baixar o corte pega mais cancelamentos, mas gera mais alarmes falsos. É só uma simulação: "
                   "o modelo em uso continua com 50%.")
        limiar = st.slider("Corte de decisão", 0.20, 0.80, float(promocao.LIMIAR_CLASSE), 0.05, format="%.2f")
    p = (prob >= limiar).astype(int)
    r = resumo_confusao(y, p)

    k = st.columns(3)
    k[0].metric("Acertou", pct(1 - r["taxa_erro"]), help="Reservas em que a previsão bateu com o que aconteceu")
    k[1].metric("Cancelamentos que passaram", pct(r["taxa_falso_negativo"]),
                help="Das reservas canceladas, quantas o modelo achou que viriam")
    k[2].metric("Alarmes falsos", pct(r["taxa_falso_positivo"]),
                help="Das reservas que vieram, quantas o modelo marcou como risco alto")

    e1, e2 = st.columns([1, 1], gap="large")
    with e1:
        mat = matriz_confusao(y, p).reset_index().melt(id_vars="real", var_name="previsto", value_name="n")
        mat["real"] = mat["real"].map({"não cancelou": "veio", "cancelou": "cancelou"})
        mat["previsto"] = mat["previsto"].map({"não cancelou": "previu: vem", "cancelou": "previu: cancela"})
        mat["rotulo"] = mat["n"].map(lambda v: f"{v:,}".replace(",", "."))
        base = alt.Chart(mat).encode(x=alt.X("previsto:N", title=None, sort=["previu: vem", "previu: cancela"],
                                             axis=alt.Axis(labelAngle=0, labelFontSize=13, orient="top")),
                                     y=alt.Y("real:N", title=None, sort=["veio", "cancelou"], axis=alt.Axis(labelFontSize=13)))
        st.altair_chart(
            (base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
             + base.mark_text(fontSize=26, fontWeight="bold").encode(
                 text="rotulo:N",
                 color=alt.condition(alt.datum.n > mat["n"].max() / 2, alt.value("white"), alt.value("black"))))
            .properties(height=260), width="stretch")
    with e2:
        sem = prev.assign(semana=pd.to_datetime(prev["arrival_date"]).dt.to_period("W").dt.start_time)
        sem = sem.groupby("semana").agg(**{"aconteceu": ("is_canceled", "mean"), "previsto": ("prob_cancelamento", "mean")})
        sem = sem.reset_index().melt("semana", var_name="série", value_name="taxa")
        st.altair_chart(alt.Chart(sem).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("semana:T", title=None, axis=alt.Axis(format="%d/%m")), y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Cancelamento",
                                                     scale=alt.Scale(domain=[0, 0.8])),
            color=alt.Color("série:N", scale=alt.Scale(range=[VERMELHO, AZUL]), legend=alt.Legend(orient="top", title=None)),
        ).properties(height=260), width="stretch")
        st.caption("Por semana: cancelamento previsto × o que aconteceu.")
    frase("O erro mais comum é deixar passar um cancelamento: o modelo é cauteloso ao dizer \"vai cancelar\".")

    with st.expander("Detalhes técnicos"):
        st.dataframe(resumo.rename(columns={"f1": "F1 macro", "acertos": "acertos"}), hide_index=True, width="stretch",
                     column_config={"acertos": st.column_config.NumberColumn(format="percent"),
                                    "F1 macro": st.column_config.NumberColumn(format="%.3f")})
        grupo = st.selectbox("Onde erra, por", ["market_segment", "deposit_type", "hotel", "customer_type",
                                                 "distribution_channel", "faixa de antecedência"])
        serie = (faixa_lead_time(prev["lead_time"], cfg()["analise_erros"]["faixas_lead_time"]).rename("faixa_lead_time")
                 if grupo.startswith("faixa") else prev[grupo])
        tab = erros_por_grupo(serie, y, p, r["falsos_positivos"] + r["falsos_negativos"], 50)
        st.dataframe(tab.drop(columns="grupo"), hide_index=True, width="stretch", column_config={
            c_: st.column_config.NumberColumn(format="percent")
            for c_ in ["taxa_cancelamento_real", "taxa_cancelamento_prevista", "taxa_erro", "parcela_dos_erros"]})


def pagina_escolha() -> None:
    st.title("Como o modelo é escolhido")
    regra = cfg()["promocao"]
    st.markdown(f"#### A cada mês, dois modelos são treinados e o melhor desafia o modelo em uso. "
                f"**Ele só entra se for melhor** (ganho mínimo de {num(regra['min_melhoria_abs'], 3)}, regra "
                f"definida antes de treinar).")

    hist = manifesto.get("historico", [])
    lotes = sorted(todos_os_lotes()["lote_id"].unique())
    linhas = []
    for h in hist:
        n = int(h["data_version"][2:5])
        linhas.append({"Mês novo": mes_pt(lotes[n - 1]), "Novo modelo": h.get("valor_vencedor"),
                       "Modelo em uso": h.get("valor_vigente"),
                       "Decisão": "trocou" if h["decisao"] == "promovido" else "manteve"})
    tab = pd.DataFrame(linhas)
    tabela_vis = tab.assign(**{c_: tab[c_].map(lambda v: "—" if pd.isna(v) else num(v, 3))
                               for c_ in ["Novo modelo", "Modelo em uso"]})

    e1, e2 = st.columns([1, 1], gap="large")
    with e1:
        st.dataframe(tabela_vis, hide_index=True, width="stretch", height=330)
        st.caption("Nota (F1) de cada um nas mesmas reservas de teste. No 1º mês ainda não havia modelo em uso.")
    with e2:
        longo = tab.melt(id_vars=["Mês novo", "Decisão"], value_vars=["Novo modelo", "Modelo em uso"],
                         var_name="série", value_name="nota").dropna()
        st.altair_chart(alt.Chart(longo).mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=3).encode(
            x=alt.X("Mês novo:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("nota:Q", scale=alt.Scale(domain=[0.70, 0.77]), title="Nota no teste (F1)"),
            color=alt.Color("série:N", scale=alt.Scale(range=[AZUL, LARANJA]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Mês novo", "série", alt.Tooltip("nota:Q", format=".3f"), "Decisão"],
        ).properties(height=330), width="stretch")
    trocas = sum(h["decisao"] == "promovido" for h in hist)
    frase(f"Em {len(hist)} meses, trocou {trocas} vezes e manteve {len(hist) - trocas}. Em mai/2017 o novo ganhou por "
          "pouco (0,003) e mesmo assim o atual ficou: a regra evita trocar por acaso.")

    with st.expander("Detalhes técnicos: competição e escolha das variáveis"):
        runs = pd.read_csv(EVID / "mlflow" / "historico_runs.csv")
        cv = runs[runs["etapa"].isin(["03_modelo_A", "03_modelo_B"])].copy()
        cv["modelo"] = cv["etapa"].map({"03_modelo_A": "A · Regressão logística", "03_modelo_B": "B · Random Forest"})
        cv["ciclo"] = cv["data_version"].str[:5]
        st.markdown("**Competição:** nota média (F1) na validação cruzada. O Random Forest venceu em todos os ciclos.")
        st.altair_chart(alt.Chart(cv).mark_bar().encode(
            x=alt.X("ciclo:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="modelo:N",
            y=alt.Y("metrics.cv_f1_macro:Q", title="F1 macro (CV)", scale=alt.Scale(domain=[0.6, 0.85])),
            color=alt.Color("modelo:N", scale=alt.Scale(range=[AZUL, LARANJA]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["modelo", alt.Tooltip("metrics.cv_f1_macro:Q", format=".3f")],
        ).properties(height=240), width="stretch")
        votos = pd.read_csv(sorted((EVID / "votacao").glob("votos_*.csv"))[-1])
        st.markdown(f"**Escolha das variáveis** (último ciclo): três métodos (SelectKBest, Random Forest e RFE) votam "
                    f"nas 20 melhores; entra quem tem 2 votos ou mais. Foram eleitas {int(votos['eleita'].sum())} de "
                    f"{len(votos)}.")
        st.dataframe(votos[votos["eleita"] == 1][["feature", "votos", "voto_selectkbest", "voto_random_forest", "voto_rfe"]],
                     hide_index=True, width="stretch")


def pagina_alerta() -> None:
    st.title("Alerta quando os dados mudam")
    st.markdown("#### Cada mês novo é comparado com **o mesmo mês do ano anterior**. Se o perfil das reservas mudar "
                "muito, o pipeline avisa: o modelo pode estar ficando desatualizado.")
    hist = pd.read_csv(sorted((EVID / "drift").glob("historico_*.csv"))[-1]).dropna(subset=["taxa_atual"])
    hist = hist[hist["referencia_tipo"] == "sazonal"]   # só os meses que têm o mesmo mês do ano anterior
    hist["mês"] = hist["lote"].map(mes_pt)
    hist["alerta"] = hist["status"].eq("alerta")
    longo = hist.melt(id_vars=["lote", "mês", "alerta"], value_vars=["taxa_referencia", "taxa_atual"],
                      var_name="série", value_name="taxa")
    longo["série"] = longo["série"].map({"taxa_referencia": "mesmo mês, ano anterior", "taxa_atual": "mês atual"})
    ordem_meses = hist["lote"].tolist()
    linhas = alt.Chart(longo).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("mês:N", title=None, sort=[mes_pt(l) for l in ordem_meses], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="Cancelamento", scale=alt.Scale(zero=False)),
        color=alt.Color("série:N", scale=alt.Scale(domain=["mês atual", "mesmo mês, ano anterior"], range=[AZUL, "#9e9e9e"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["mês", "série", alt.Tooltip("taxa:Q", format=".0%")])
    alertas = alt.Chart(hist[hist["alerta"]]).mark_point(size=260, color=VERMELHO, strokeWidth=3).encode(
        x=alt.X("mês:N", sort=[mes_pt(l) for l in ordem_meses]), y="taxa_atual:Q",
        tooltip=["mês", alt.Tooltip("diferenca_pp:Q", title="variação (p.p.)")])
    st.altair_chart((linhas + alertas).properties(height=340), width="stretch")
    n_alerta = int(hist["alerta"].sum())
    frase(f"Os círculos vermelhos são os {n_alerta} meses com alerta. Em 2017 o cancelamento ficou acima do de 2016, "
          "e o pipeline avisou.")

    with st.expander("Detalhes técnicos: o que mudou em cada mês"):
        df = todos_os_lotes()
        lotes = sorted(df["lote_id"].unique())
        atual = st.select_slider("Mês", options=lotes, value="lote_2017-04" if "lote_2017-04" in lotes else lotes[-1],
                                 format_func=mes_pt)
        tabela, resumo = monitorar_lote(df, atual, cfg(), lotes)
        if resumo["status"] == "sem_referencia":
            st.info("Primeiro mês: ainda não há com o que comparar.")
            return
        c = cfg()["drift"]
        st.caption(f"PSI por variável: abaixo de {num(c['psi']['atencao'])} estável, acima de {num(c['psi']['alerta'])} "
                   f"alerta. `{', '.join(c['informativas'])}` aparece só como informação (a carteira de agências se "
                   "renova todo ano).")
        st.altair_chart(alt.Chart(tabela).mark_bar().encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("psi:Q", title="PSI"),
            color=alt.Color("status:N", scale=alt.Scale(domain=["ok", "atencao", "alerta"],
                                                        range=[VERDE, "#fbc02d", VERMELHO]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["feature", alt.Tooltip("psi:Q", format=".3f"), "status"],
        ).properties(height=max(260, 16 * len(tabela))), width="stretch")


{
    "projeto": pagina_projeto,
    "dados": pagina_dados,
    "testar": pagina_testar,
    "acertou": pagina_acertou,
    "escolha": pagina_escolha,
    "alerta": pagina_alerta,
}[pagina]()
