"""Matriz de confusão e análise de erros no teste do ciclo (notebook 04, feature F8).

Responde "onde o modelo erra": por hotel, segmento de mercado, tipo de depósito, tipo de cliente, canal e
faixa de antecedência (``lead_time``). Parâmetros em ``cfg["analise_erros"]``.

Só descreve o modelo. Nada daqui entra na eleição, na regra de promoção ou nas métricas da decisão
(critério 8): a análise usa o mesmo teste e o mesmo limiar (``src.promocao.LIMIAR_CLASSE``) que o
notebook 04 já usa para comparar vencedor e vigente.

Saídas (``analise_erros.dir_evidencias``, padrão ``evidencias/promocao/``):
  * ``erros_<cycle_id>.json``: matriz, contagens, taxas e os grupos com mais erro;
  * ``erros_por_grupo_<cycle_id>.csv``: uma linha por valor de cada grupo;
  * ``matriz_confusao_<cycle_id>.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAIZ, resolver
from src.preprocessamento import lista_features
from src.promocao import LIMIAR_CLASSE

ROTULOS = {0: "não cancelou", 1: "cancelou"}
COLUNAS_GRUPO = [
    "grupo", "valor", "n", "taxa_cancelamento_real", "taxa_cancelamento_prevista",
    "falsos_positivos", "falsos_negativos", "taxa_erro", "parcela_dos_erros",
]


def matriz_confusao(y_true, y_pred) -> pd.DataFrame:
    """Matriz 2x2: linhas = desfecho real, colunas = previsão."""
    real = pd.Series(np.asarray(y_true, dtype=int)).map(ROTULOS)
    prev = pd.Series(np.asarray(y_pred, dtype=int)).map(ROTULOS)
    ordem = list(ROTULOS.values())
    tabela = pd.crosstab(real, prev).reindex(index=ordem, columns=ordem, fill_value=0)
    tabela.index.name, tabela.columns.name = "real", "previsto"
    return tabela


def resumo_confusao(y_true, y_pred) -> dict:
    """Contagens e taxas de erro, com nomes que o hotel entende."""
    m = matriz_confusao(y_true, y_pred).to_numpy()
    tn, fp, fn, tp = (int(v) for v in m.ravel())
    n = tn + fp + fn + tp
    return {
        "n": n,
        "verdadeiros_negativos": tn,
        "falsos_positivos": fp,      # previu cancelamento, mas o hóspede veio
        "falsos_negativos": fn,      # previu que viria, mas cancelou
        "verdadeiros_positivos": tp,
        "taxa_erro": (fp + fn) / n if n else 0.0,
        "taxa_falso_positivo": fp / (tn + fp) if tn + fp else 0.0,
        "taxa_falso_negativo": fn / (fn + tp) if fn + tp else 0.0,
    }


def faixa_lead_time(lead_time: pd.Series, cortes: list[float]) -> pd.Series:
    """Faixas de antecedência em dias, ex.: ``[0, 7, 30]`` -> ``0-6``, ``7-29``, ``30+``."""
    cortes = sorted(cortes)
    rotulos = [f"{a:g}-{b - 1:g}" for a, b in zip(cortes[:-1], cortes[1:])] + [f"{cortes[-1]:g}+"]
    faixas = pd.cut(lead_time, bins=[*cortes, np.inf], right=False, labels=rotulos)
    return faixas.astype(str)


def erros_por_grupo(grupos: pd.Series, y_true, y_pred, total_erros: int, min_reservas: int = 0) -> pd.DataFrame:
    """Uma linha por valor do grupo. Valores com menos de ``min_reservas`` viram ``outros``."""
    g = grupos.astype(str).reset_index(drop=True)
    contagem = g.value_counts()
    g = g.where(g.map(contagem) >= min_reservas, "outros")
    df = pd.DataFrame({"valor": g, "y": np.asarray(y_true, dtype=int), "p": np.asarray(y_pred, dtype=int)})
    df["fp"] = (df["p"] == 1) & (df["y"] == 0)
    df["fn"] = (df["p"] == 0) & (df["y"] == 1)
    tabela = df.groupby("valor").agg(
        n=("y", "size"),
        taxa_cancelamento_real=("y", "mean"),
        taxa_cancelamento_prevista=("p", "mean"),
        falsos_positivos=("fp", "sum"),
        falsos_negativos=("fn", "sum"),
    ).reset_index()
    erros = tabela["falsos_positivos"] + tabela["falsos_negativos"]
    tabela["taxa_erro"] = erros / tabela["n"]
    tabela["parcela_dos_erros"] = erros / total_erros if total_erros else 0.0
    tabela.insert(0, "grupo", grupos.name)
    return tabela[COLUNAS_GRUPO].sort_values("taxa_erro", ascending=False).reset_index(drop=True)


def analisar_erros(modelo, teste: pd.DataFrame, cfg: dict) -> dict:
    """Matriz de confusão, taxas de erro e erros por grupo do ``modelo`` no teste do ciclo."""
    a = cfg["analise_erros"]
    alvo = cfg["avaliacao"]["alvo"]
    y_true = teste[alvo].to_numpy(dtype=int)
    y_pred = (modelo.predict_proba(teste[lista_features(cfg)])[:, 1] >= LIMIAR_CLASSE).astype(int)

    resumo = resumo_confusao(y_true, y_pred)
    total_erros = resumo["falsos_positivos"] + resumo["falsos_negativos"]
    colunas = {c: teste[c] for c in a["grupos"]}
    colunas["faixa_lead_time"] = faixa_lead_time(teste["lead_time"], a["faixas_lead_time"]).rename("faixa_lead_time")
    por_grupo = pd.concat(
        [erros_por_grupo(s.rename(nome), y_true, y_pred, total_erros, a["min_reservas_grupo"]) for nome, s in colunas.items()],
        ignore_index=True,
    )
    relevantes = por_grupo[(por_grupo["valor"] != "outros") & (por_grupo["n"] >= a["min_reservas_grupo"])]
    return {
        "limiar": LIMIAR_CLASSE,
        "matriz": matriz_confusao(y_true, y_pred),
        "resumo": resumo,
        "por_grupo": por_grupo,
        "maiores_taxas_erro": relevantes.nlargest(a["n_destaques"], "taxa_erro").reset_index(drop=True),
    }


def _grafico_matriz(matriz: pd.DataFrame, caminho: Path, titulo: str) -> None:
    from matplotlib.figure import Figure   # sem pyplot, como em src.drift

    valores = matriz.to_numpy()
    fig = Figure(figsize=(4.6, 3.8))
    ax = fig.subplots()
    ax.imshow(valores, cmap="Blues")
    for (i, j), v in np.ndenumerate(valores):
        ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                color="white" if v > valores.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], [f"previsto: {c}" for c in matriz.columns], fontsize=8)
    ax.set_yticks([0, 1], [f"real: {c}" for c in matriz.index], fontsize=8)
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)


def salvar_analise_erros(analise: dict, ctx: dict, cfg: dict, modelo_analisado: str, raiz: Path = RAIZ) -> dict[str, Path]:
    """Grava a análise em ``analise_erros.dir_evidencias``, identificada pelo ``cycle_id``."""
    destino = resolver(cfg["analise_erros"]["dir_evidencias"], raiz)
    destino.mkdir(parents=True, exist_ok=True)
    cid = ctx["cycle_id"]
    caminhos = {
        "json": destino / f"erros_{cid}.json",
        "csv": destino / f"erros_por_grupo_{cid}.csv",
        "png": destino / f"matriz_confusao_{cid}.png",
    }
    registro = {
        "cycle_id": cid,
        "data_version": ctx["data_version"],
        "modelo_analisado": modelo_analisado,
        "limiar": analise["limiar"],
        "matriz_confusao": analise["matriz"].to_dict(orient="index"),
        **analise["resumo"],
        "maiores_taxas_erro": analise["maiores_taxas_erro"].to_dict(orient="records"),
    }
    caminhos["json"].write_text(json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    analise["por_grupo"].assign(cycle_id=cid, data_version=ctx["data_version"]).to_csv(caminhos["csv"], index=False)
    _grafico_matriz(analise["matriz"], caminhos["png"], f"Teste do ciclo {ctx['data_version']}\nmodelo em uso: {modelo_analisado}")
    return caminhos


def metricas_para_mlflow(analise: dict) -> dict[str, float]:
    """Contagens e taxas da matriz, com prefixo ``teste_em_uso_`` (o modelo que segue vigente)."""
    return {f"teste_em_uso_{k}": float(v) for k, v in analise["resumo"].items()}
