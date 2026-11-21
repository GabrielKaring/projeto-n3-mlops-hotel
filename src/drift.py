"""Feature F10 (critério 10, recurso adicional): monitoramento de drift com alerta.

Dono: ver ``config/equipe.yaml``. Problema, justificativa e regra: ``docs/f10_drift.md``.

Problema: a cada ciclo chega um lote mensal novo, e o modelo vigente foi treinado com o passado. Se o
perfil das reservas ou a taxa de cancelamento mudam, o modelo pode piorar sem que ninguém perceba até a
próxima promoção. Este módulo compara o lote mais recente com uma referência e avisa quando a mudança passa
dos limiares fixados em ``cfg["drift"]``.

Contrato de entrada:
  * lotes registrados em ``data/raw/manifest.json`` e limpos por ``limpar_e_engenhar`` (feature F2);
  * regra em ``cfg["drift"]``, fixada antes de rodar nos dados reais;
  * ``data/processed/ciclo_atual.json`` (notebook 01), para usar o mesmo ``cycle_id`` e ``data_version``
    do ciclo. Sem esse arquivo (execução avulsa), gera um ``cycle_id`` próprio.

Contrato de saída (``evidencias/drift/``):
  * ``drift_<cycle_id>.json``: status geral, drift do alvo, PSI e status por feature, referência e limiares;
  * ``drift_<cycle_id>.csv`` e ``drift_<cycle_id>.png``: a tabela e o gráfico do PSI por feature;
  * com ``--historico``: ``historico_<cycle_id>.csv`` e ``.png``, um lote por linha;
  * alerta no log. No GitHub Actions, ``::warning`` na execução e um resumo em ``$GITHUB_STEP_SUMMARY``.

O monitoramento só lê dados. Nada do que ele calcula entra na seleção de features, no treino ou na
promoção (critérios 3 a 8): é um aviso para quem opera o modelo, e não muda nenhuma decisão automática.

Uso: ``python -m src.drift [--lote lote_2017-04] [--historico] [--falhar-em-alerta]``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAIZ, carregar_config, carregar_contexto, resolver
from src.ingestao import (
    calcular_data_version,
    carregar_manifesto,
    consolidar_lotes,
    gerar_cycle_id,
    verificar_integridade,
)
from src.preprocessamento import lista_features, limpar_e_engenhar
from src.schema import ALVO

EPS = 1e-4                       # piso das proporções: evita log(0) quando uma faixa fica vazia
STATUS = ("ok", "atencao", "alerta")   # em ordem de gravidade
RARAS = "<rara>"
_PERIODO = re.compile(r"(\d{4})-(\d{2})$")
COLUNAS_TABELA = ["feature", "tipo", "psi", "status"]

# --------------------------------------------------------------------------- PSI


def _proporcoes(contagens: np.ndarray) -> np.ndarray:
    total = contagens.sum()
    p = contagens / total if total else np.zeros_like(contagens, dtype=float)
    return np.clip(p, EPS, None)


def _psi(p_ref: np.ndarray, p_atual: np.ndarray) -> float:
    return float(np.sum((p_atual - p_ref) * np.log(p_atual / p_ref)))


def psi_categorica(ref: pd.Series, atual: pd.Series, min_frequencia: float = 0.0) -> float:
    """PSI entre duas amostras de uma variável categórica (ou numérica com poucos valores).

    Categorias com frequência abaixo de ``min_frequencia`` na referência viram uma só (``<rara>``), nos dois
    lados. Uma categoria nova que aparece em massa no lote atual cai nesse grupo e continua sendo detectada.
    """
    r = ref.astype(str)
    a = atual.astype(str)
    if min_frequencia > 0:
        freq = r.value_counts(normalize=True)
        frequentes = set(freq[freq >= min_frequencia].index)
        r = r.where(r.isin(frequentes), RARAS)
        a = a.where(a.isin(frequentes), RARAS)
    cont_r, cont_a = r.value_counts(), a.value_counts()
    categorias = cont_r.index.union(cont_a.index)
    p_ref = _proporcoes(cont_r.reindex(categorias, fill_value=0).to_numpy(dtype=float))
    p_atual = _proporcoes(cont_a.reindex(categorias, fill_value=0).to_numpy(dtype=float))
    return _psi(p_ref, p_atual)


def psi_numerica(ref: pd.Series, atual: pd.Series, n_bins: int = 10) -> float:
    """PSI de uma variável contínua, com faixas definidas pelos quantis da referência.

    As faixas das pontas são abertas (-inf e +inf), então valores do lote atual fora do intervalo da
    referência contam nas faixas extremas. Ausentes formam uma faixa própria.
    """
    if n_bins < 2:
        raise ValueError("n_bins deve ser pelo menos 2")
    r = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(atual, errors="coerce").to_numpy(dtype=float)
    validos = r[~np.isnan(r)]
    if validos.size == 0:
        raise ValueError("A referência não tem valores numéricos válidos")
    cortes = np.unique(np.quantile(validos, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def contar(v: np.ndarray) -> np.ndarray:
        ausente = np.isnan(v)
        faixa = np.digitize(v[~ausente], cortes, right=True)   # 0 .. len(cortes)
        cont = np.bincount(faixa, minlength=len(cortes) + 1).astype(float)
        return np.append(cont, ausente.sum())

    return _psi(_proporcoes(contar(r)), _proporcoes(contar(a)))


def classificar(valor: float, atencao: float, alerta: float) -> str:
    if valor >= alerta:
        return "alerta"
    if valor >= atencao:
        return "atencao"
    return "ok"


def _tipo(coluna: str, ref: pd.Series, categoricas: set[str], max_discretos: int) -> str:
    if coluna in categoricas or not pd.api.types.is_numeric_dtype(ref):
        return "categorica"
    if ref.nunique(dropna=True) <= max_discretos:
        return "discreta"
    return "numerica"


# --------------------------------------------------------------------------- regra do ciclo


def colunas_monitoradas(cfg: dict) -> list[str]:
    """Features de entrada do modelo, menos as de calendário listadas em ``drift.ignorar``."""
    ignorar = set(cfg["drift"].get("ignorar", []))
    return [c for c in lista_features(cfg) if c not in ignorar]


def drift_alvo(ref: pd.DataFrame, atual: pd.DataFrame, cfg: dict) -> dict:
    """Variação da taxa de cancelamento, em pontos percentuais."""
    limites = cfg["drift"]["alvo"]
    taxa_ref, taxa_atual = float(ref[ALVO].mean()), float(atual[ALVO].mean())
    diferenca = (taxa_atual - taxa_ref) * 100
    return {
        "taxa_referencia": round(taxa_ref, 4),
        "taxa_atual": round(taxa_atual, 4),
        "diferenca_pp": round(diferenca, 2),
        "status": classificar(abs(diferenca), limites["atencao_pp"], limites["alerta_pp"]),
    }


def avaliar_drift(df_ref: pd.DataFrame, df_atual: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Compara duas amostras já limpas pelo ETL. Devolve ``(tabela por feature, resumo)``.

    O status geral é o mais grave entre as features e o alvo. Se uma das amostras tem menos de
    ``drift.min_linhas`` linhas, o status é ``amostra_insuficiente`` e não há alerta.
    """
    d = cfg["drift"]
    if df_ref.empty or df_atual.empty:
        raise ValueError("Referência e lote atual precisam ter linhas depois do ETL")
    colunas = colunas_monitoradas(cfg)
    faltando = [c for c in colunas + [ALVO] if c not in df_ref.columns or c not in df_atual.columns]
    if faltando:
        raise ValueError(f"Colunas monitoradas ausentes: {faltando}")

    categoricas = set(cfg["features"]["categoricas"])
    linhas = []
    for col in colunas:
        tipo = _tipo(col, df_ref[col], categoricas, d["max_valores_discretos"])
        if tipo == "numerica":
            psi = psi_numerica(df_ref[col], df_atual[col], d["n_bins"])
        else:
            psi = psi_categorica(df_ref[col], df_atual[col], d.get("min_frequencia_categoria", 0.0))
        status = classificar(psi, d["psi"]["atencao"], d["psi"]["alerta"])
        linhas.append({"feature": col, "tipo": tipo, "psi": round(psi, 4), "status": status})
    tabela = (
        pd.DataFrame(linhas, columns=COLUNAS_TABELA)
        .sort_values(["psi", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    alvo = drift_alvo(df_ref, df_atual, cfg)
    n_ref, n_atual = int(len(df_ref)), int(len(df_atual))
    if min(n_ref, n_atual) < d["min_linhas"]:
        status = "amostra_insuficiente"
    else:
        status = max(list(tabela["status"]) + [alvo["status"]], key=STATUS.index)

    resumo = {
        "status": status,
        "n_linhas_referencia": n_ref,
        "n_linhas_atual": n_atual,
        "alvo": alvo,
        "n_features": int(len(tabela)),
        "n_features_atencao": int((tabela["status"] == "atencao").sum()),
        "n_features_alerta": int((tabela["status"] == "alerta").sum()),
        "features_alerta": tabela.loc[tabela["status"] == "alerta", "feature"].tolist(),
        "maior_psi": {"feature": str(tabela["feature"].iloc[0]), "psi": float(tabela["psi"].iloc[0])},
        "limiares": {"psi": d["psi"], "alvo": d["alvo"], "min_linhas": d["min_linhas"]},
    }
    return tabela, resumo


def escolher_referencia(lotes: list[str], atual: str, cfg: dict) -> dict:
    """Lotes contra os quais o lote ``atual`` é comparado.

    ``sazonal``: o mesmo mês do ano anterior (``lote_2017-04`` -> ``lote_2016-04``), se já incorporado.
    Senão, ou com ``referencia: anteriores``, os últimos ``janela_lotes`` lotes antes do atual.
    Devolve ``{"tipo": ..., "lotes": [...]}``; a lista vem vazia para o primeiro lote.
    """
    d = cfg["drift"]
    if d["referencia"] not in ("sazonal", "anteriores"):
        raise ValueError(f"drift.referencia desconhecida: {d['referencia']}")
    if d["janela_lotes"] < 1:
        raise ValueError("drift.janela_lotes deve ser pelo menos 1")
    ordenados = sorted(lotes)
    if atual not in ordenados:
        raise ValueError(f"O lote {atual} não está entre os lotes incorporados")
    anteriores = ordenados[: ordenados.index(atual)]

    if d["referencia"] == "sazonal":
        m = _PERIODO.search(atual)
        if m:
            mesmo_mes = f"{atual[: m.start()]}{int(m.group(1)) - 1}-{m.group(2)}"
            if mesmo_mes in anteriores:
                return {"tipo": "sazonal", "lotes": [mesmo_mes]}
    return {"tipo": "anteriores", "lotes": anteriores[-d["janela_lotes"]:]}


def monitorar_lote(df: pd.DataFrame, atual: str, cfg: dict, lotes: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Avalia o drift de um lote contra a sua referência. ``df`` é a saída de ``limpar_e_engenhar``."""
    lotes = sorted(df["lote_id"].unique()) if lotes is None else lotes
    referencia = escolher_referencia(lotes, atual, cfg)
    df_atual = df[df["lote_id"] == atual]
    base = {"lote_atual": atual, "referencia": referencia}
    if not referencia["lotes"]:
        vazio = {
            "status": "sem_referencia",
            "n_linhas_referencia": 0,
            "n_linhas_atual": int(len(df_atual)),
            "alvo": None,
            "n_features": 0,
            "n_features_atencao": 0,
            "n_features_alerta": 0,
            "features_alerta": [],
            "maior_psi": None,
        }
        return pd.DataFrame(columns=COLUNAS_TABELA), {**base, **vazio}
    df_ref = df[df["lote_id"].isin(referencia["lotes"])]
    tabela, resumo = avaliar_drift(df_ref, df_atual, cfg)
    return tabela, {**base, **resumo}


def historico_drift(df: pd.DataFrame, cfg: dict, lotes: list[str] | None = None) -> pd.DataFrame:
    """Uma linha por lote incorporado, cada um contra a sua própria referência. Serve de evidência."""
    lotes = sorted(df["lote_id"].unique()) if lotes is None else sorted(lotes)
    linhas = []
    for lote in lotes:
        _, r = monitorar_lote(df, lote, cfg, lotes)
        alvo = r["alvo"] or {}
        maior = r["maior_psi"] or {}
        linhas.append({
            "lote": lote,
            "referencia_tipo": r["referencia"]["tipo"],
            "referencia": ";".join(r["referencia"]["lotes"]),
            "n_linhas_referencia": r["n_linhas_referencia"],
            "n_linhas_atual": r["n_linhas_atual"],
            "taxa_referencia": alvo.get("taxa_referencia"),
            "taxa_atual": alvo.get("taxa_atual"),
            "diferenca_pp": alvo.get("diferenca_pp"),
            "status_alvo": alvo.get("status"),
            "maior_psi_feature": maior.get("feature"),
            "maior_psi": maior.get("psi"),
            "n_features_alerta": r["n_features_alerta"],
            "features_alerta": ";".join(r["features_alerta"]),
            "status": r["status"],
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- dados e contexto do ciclo


def carregar_dados_monitorados(cfg: dict, raiz: Path = RAIZ) -> tuple[pd.DataFrame, dict]:
    """Consolida os lotes do manifesto (conferindo o sha256) e aplica o mesmo ETL da F2."""
    dir_raw = resolver(cfg["dados"]["dir_raw"], raiz)
    manifesto = carregar_manifesto(resolver(cfg["dados"]["manifesto"], raiz))
    verificar_integridade(dir_raw, manifesto)
    df, _ = limpar_e_engenhar(consolidar_lotes(dir_raw, manifesto), cfg)
    return df, manifesto


def contexto_do_ciclo(cfg: dict, manifesto: dict, raiz: Path = RAIZ, ambiente: dict | None = None) -> dict:
    """``cycle_id`` e ``data_version`` do ciclo em curso, lidos do ``ciclo_atual.json`` do notebook 01.

    Falha se o contexto for de outra versão dos dados (o notebook 01 não rodou depois do lote novo).
    """
    data_version = calcular_data_version(manifesto)
    try:
        ctx = carregar_contexto(cfg, raiz)
    except FileNotFoundError:
        return {"cycle_id": gerar_cycle_id(ambiente=ambiente), "data_version": data_version, "origem_contexto": "avulso"}
    if ctx.get("data_version") != data_version:
        raise ValueError(
            f"ciclo_atual.json é da versão {ctx.get('data_version')}, mas os lotes estão em {data_version}. "
            "Execute o notebook 01_etl antes do monitoramento de drift."
        )
    return {"cycle_id": ctx["cycle_id"], "data_version": data_version, "origem_contexto": "ciclo_atual.json"}


# --------------------------------------------------------------------------- evidências


def _grafico_psi(tabela: pd.DataFrame, cfg: dict, caminho: Path, titulo: str) -> None:
    from matplotlib.figure import Figure   # API sem pyplot: não mexe no backend de quem importar o módulo

    limites = cfg["drift"]["psi"]
    cores = {"ok": "#4C9F70", "atencao": "#E0A100", "alerta": "#C0392B"}
    t = tabela.iloc[::-1]
    fig = Figure(figsize=(8, max(3.0, 0.28 * len(t) + 1.2)))
    ax = fig.subplots()
    ax.barh(t["feature"], t["psi"], color=[cores[s] for s in t["status"]])
    ax.axvline(limites["atencao"], ls="--", lw=1, color=cores["atencao"], label=f"atenção ({limites['atencao']})")
    ax.axvline(limites["alerta"], ls="--", lw=1, color=cores["alerta"], label=f"alerta ({limites['alerta']})")
    ax.set_xlabel("PSI (lote atual × referência)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)


def _grafico_historico(hist: pd.DataFrame, caminho: Path, titulo: str) -> None:
    from matplotlib.figure import Figure

    h = hist[hist["taxa_atual"].notna()]
    if h.empty:
        return
    x = np.arange(len(h))
    fig = Figure(figsize=(max(6.0, 0.45 * len(h) + 2), 4))
    ax = fig.subplots()
    ax.plot(x, h["taxa_referencia"] * 100, marker="o", lw=1, color="#7F8C8D", label="referência")
    ax.plot(x, h["taxa_atual"] * 100, marker="o", lw=1.5, color="#2E86C1", label="lote")
    alerta = (h["status"] == "alerta").to_numpy()
    ax.scatter(x[alerta], h["taxa_atual"].to_numpy()[alerta] * 100, s=120, facecolors="none",
               edgecolors="#C0392B", linewidths=2, label="status geral = alerta", zorder=3)
    ax.set_xticks(x, h["lote"].str.replace("lote_", "", regex=False), rotation=60, fontsize=8)
    ax.set_ylabel("taxa de cancelamento (%)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)


def salvar_drift(tabela: pd.DataFrame, resumo: dict, ctx: dict, cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    """Grava o resultado do ciclo em ``evidencias/drift/``, identificado pelo ``cycle_id``."""
    destino = resolver(cfg["drift"]["dir_evidencias"], raiz)
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / f"drift_{ctx['cycle_id']}"
    registro = {
        "cycle_id": ctx["cycle_id"],
        "data_version": ctx["data_version"],
        "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **resumo,
        "features": tabela.to_dict(orient="records"),
    }
    caminhos = {"json": base.with_suffix(".json"), "csv": base.with_suffix(".csv")}
    caminhos["json"].write_text(json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tabela.assign(cycle_id=ctx["cycle_id"], data_version=ctx["data_version"], lote_atual=resumo["lote_atual"]).to_csv(
        caminhos["csv"], index=False
    )
    if not tabela.empty:
        caminhos["png"] = base.with_suffix(".png")
        ref = ", ".join(resumo["referencia"]["lotes"])
        _grafico_psi(tabela, cfg, caminhos["png"], f"{resumo['lote_atual']} × {ref} · status: {resumo['status']}")
    return caminhos


def salvar_historico(hist: pd.DataFrame, ctx: dict, cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    destino = resolver(cfg["drift"]["dir_evidencias"], raiz)
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / f"historico_{ctx['cycle_id']}"
    caminhos = {"historico_csv": base.with_suffix(".csv")}
    hist.assign(cycle_id=ctx["cycle_id"], data_version=ctx["data_version"]).to_csv(caminhos["historico_csv"], index=False)
    if hist["taxa_atual"].notna().any():
        caminhos["historico_png"] = base.with_suffix(".png")
        _grafico_historico(hist, caminhos["historico_png"], f"Taxa de cancelamento por lote × referência ({ctx['data_version']})")
    return caminhos


# --------------------------------------------------------------------------- alerta


def mensagem_alerta(resumo: dict) -> str:
    ref = ", ".join(resumo["referencia"]["lotes"]) or "nenhuma"
    partes = [f"{resumo['lote_atual']} × {ref} ({resumo['referencia']['tipo']}): {resumo['status'].upper()}"]
    alvo = resumo.get("alvo")
    if alvo:
        partes.append(
            f"taxa de cancelamento {alvo['taxa_referencia'] * 100:.1f}% -> {alvo['taxa_atual'] * 100:.1f}% "
            f"({alvo['diferenca_pp']:+.1f} p.p., {alvo['status']})"
        )
    if resumo.get("features_alerta"):
        partes.append("PSI no nível de alerta em: " + ", ".join(resumo["features_alerta"]))
    return "; ".join(partes)


def _resumo_markdown(resumo: dict, tabela: pd.DataFrame, ctx: dict) -> str:
    ref = ", ".join(resumo["referencia"]["lotes"]) or "nenhuma"
    linhas = [
        f"### Drift de dados · {resumo['lote_atual']}",
        "",
        f"Status: **{resumo['status'].upper()}** · referência: {ref} ({resumo['referencia']['tipo']}) · "
        f"`cycle_id` {ctx['cycle_id']} · `data_version` {ctx['data_version']}",
        "",
    ]
    alvo = resumo.get("alvo")
    if alvo:
        linhas += [
            "| | Referência | Lote atual | Diferença | Status |",
            "|---|---|---|---|---|",
            f"| Taxa de cancelamento | {alvo['taxa_referencia'] * 100:.1f}% | {alvo['taxa_atual'] * 100:.1f}% "
            f"| {alvo['diferenca_pp']:+.1f} p.p. | {alvo['status']} |",
            "",
        ]
    if not tabela.empty:
        linhas += ["| Feature | Tipo | PSI | Status |", "|---|---|---|---|"]
        linhas += [f"| {r.feature} | {r.tipo} | {r.psi:.3f} | {r.status} |" for r in tabela.head(10).itertuples()]
    return "\n".join(linhas) + "\n"


def emitir_alerta(resumo: dict, tabela: pd.DataFrame, ctx: dict, ambiente: dict | None = None) -> str:
    """Escreve o alerta no log. No GitHub Actions vira anotação da execução e resumo do job."""
    env = os.environ if ambiente is None else ambiente
    msg = mensagem_alerta(resumo)
    status = resumo["status"]
    if env.get("GITHUB_ACTIONS") == "true":
        seguro = msg.replace("%", "%25")   # % é caractere de escape nos comandos do Actions
        if status == "alerta":
            print(f"::warning title=Drift de dados::{seguro}")
        elif status != "ok":
            print(f"::notice title=Drift de dados::{seguro}")
        else:
            print(f"Drift de dados: {msg}")
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write(_resumo_markdown(resumo, tabela, ctx))
    else:
        print(("ALERTA DE DRIFT: " if status == "alerta" else "Drift de dados: ") + msg)
    return msg


# --------------------------------------------------------------------------- execução


def executar(cfg: dict, raiz: Path = RAIZ, lote: str | None = None, historico: bool = False,
             ambiente: dict | None = None) -> dict:
    """Fluxo completo: dados do ciclo -> drift do lote mais recente -> evidências -> alerta."""
    df, manifesto = carregar_dados_monitorados(cfg, raiz)
    ctx = contexto_do_ciclo(cfg, manifesto, raiz, ambiente)
    lotes = sorted(l["lote_id"] for l in manifesto["lotes"])
    atual = lote or lotes[-1]
    tabela, resumo = monitorar_lote(df, atual, cfg, lotes)
    caminhos = salvar_drift(tabela, resumo, ctx, cfg, raiz)
    hist = None
    if historico:
        hist = historico_drift(df, cfg, lotes)
        caminhos.update(salvar_historico(hist, ctx, cfg, raiz))
    emitir_alerta(resumo, tabela, ctx, ambiente)
    return {"ctx": ctx, "resumo": resumo, "tabela": tabela, "historico": hist, "caminhos": caminhos}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Monitora o drift do lote mais recente contra a referência (F10).")
    ap.add_argument("--lote", default=None, help="lote a avaliar, ex.: lote_2017-04 (padrão: o mais recente)")
    ap.add_argument("--historico", action="store_true", help="também avalia todos os lotes já incorporados")
    ap.add_argument("--falhar-em-alerta", action="store_true", help="sai com código 1 se o status geral for alerta")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    r = executar(carregar_config(args.config), lote=args.lote, historico=args.historico)
    print(json.dumps({k: str(v) for k, v in r["caminhos"].items()}, indent=2, ensure_ascii=False))
    return 1 if args.falhar_em_alerta and r["resumo"]["status"] == "alerta" else 0


if __name__ == "__main__":
    raise SystemExit(main())
