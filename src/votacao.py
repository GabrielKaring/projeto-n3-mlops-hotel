"""Feature F4 (critério 4): votação e eleição do conjunto final de features.

Dono: ver ``config/equipe.yaml``. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada:
  * ``rankings`` produzidos por ``src.features.calcular_rankings_por_fold`` (feature F3).
  * Regra de votação em ``cfg["selecao"]["votacao"]``, definida ANTES de ver resultados.

Contrato de saída (para o notebook 03, feature F5):
  * ``data/processed/features_eleitas.json`` com: cycle_id, data_version, features eleitas
    (nomes após o pré-processador), regra de votação usada e k_top.
  * ``evidencias/votacao/votos_<cycle_id>.csv`` com a tabela de votos e os rankings dos 3 métodos.
  * O registro no MLflow dos votos e do conjunto final depende de ``src/rastreio.py`` (feature F6).

Regra: cada método vota nas features que aprovou (top-k, com empates). É eleita a feature com pelo menos
``min_votos``. Se sobrarem menos de ``min_features``, o conjunto é completado pelo melhor ranking médio.
Empates são desfeitos por mais votos, depois menor ranking médio, depois nome, para o resultado ser único.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RAIZ, resolver
from src.features import listas_aprovadas

DIR_EVIDENCIAS = "evidencias/votacao"
NOME_ELEITAS = "features_eleitas.json"
DESEMPATES = ("ranking_medio",)


def votar(rankings: dict[str, pd.Series], cfg: dict) -> pd.DataFrame:
    """Monta a tabela de votos: uma linha por feature, com score, posição e voto de cada método, total e ranking médio."""
    aprovadas = listas_aprovadas(rankings, cfg["selecao"]["k_top"])
    colunas = {}
    for metodo, score in rankings.items():
        colunas[f"score_{metodo}"] = score
        colunas[f"posicao_{metodo}"] = score.rank(ascending=False, method="min").astype(int)
        colunas[f"voto_{metodo}"] = score.index.isin(aprovadas[metodo]).astype(int)
    tabela = pd.DataFrame(colunas)
    tabela["votos"] = tabela[[f"voto_{m}" for m in rankings]].sum(axis=1)
    tabela["ranking_medio"] = tabela[[f"posicao_{m}" for m in rankings]].mean(axis=1)
    tabela.index.name = "feature"
    return tabela.reset_index().sort_values(
        ["votos", "ranking_medio", "feature"], ascending=[False, True, True]
    ).set_index("feature")


def eleger_features(tabela_votos: pd.DataFrame, cfg: dict) -> list[str]:
    """Aplica a regra de ``cfg['selecao']['votacao']`` (mín. de votos, desempate, mín. de features)."""
    regra = cfg["selecao"]["votacao"]
    n_metodos = sum(c.startswith("voto_") for c in tabela_votos.columns)
    if not 1 <= regra["min_votos"] <= n_metodos:
        raise ValueError(f"min_votos deve estar entre 1 e {n_metodos}; veio {regra['min_votos']}")
    if regra["desempate"] not in DESEMPATES:
        raise ValueError(f"Desempate não suportado: {regra['desempate']}")
    if regra["min_features"] > len(tabela_votos):
        raise ValueError(f"min_features={regra['min_features']}, mas só há {len(tabela_votos)} features")
    ordenada = tabela_votos.reset_index().sort_values(
        ["votos", "ranking_medio", "feature"], ascending=[False, True, True]
    )
    n_eleitas = max(int((ordenada["votos"] >= regra["min_votos"]).sum()), regra["min_features"])
    return ordenada["feature"].head(n_eleitas).tolist()


def salvar_votacao(tabela_votos: pd.DataFrame, eleitas: list[str], ctx: dict, cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    """Grava a tabela de votos em ``evidencias/votacao/`` e o conjunto eleito para o notebook 03."""
    identificacao = {"cycle_id": ctx["cycle_id"], "data_version": ctx["data_version"]}
    destino_votos = resolver(DIR_EVIDENCIAS, raiz)
    destino_votos.mkdir(parents=True, exist_ok=True)
    caminho_votos = destino_votos / f"votos_{ctx['cycle_id']}.csv"
    tabela_votos.assign(eleita=tabela_votos.index.isin(eleitas).astype(int), **identificacao).to_csv(
        caminho_votos, float_format="%.10f"
    )

    caminho_eleitas = resolver(cfg["dados"]["dir_processed"], raiz) / NOME_ELEITAS
    caminho_eleitas.parent.mkdir(parents=True, exist_ok=True)
    caminho_eleitas.write_text(json.dumps({
        **identificacao,
        "features_eleitas": eleitas,
        "n_features": len(eleitas),
        "k_top": cfg["selecao"]["k_top"],
        "regra_votacao": cfg["selecao"]["votacao"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"votos": caminho_votos, "eleitas": caminho_eleitas}
