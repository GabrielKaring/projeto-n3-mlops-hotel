"""Feature F3 (critério 3): três métodos de seleção de features.

Dono: ver ``config/equipe.yaml``. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada (notebook 02, seção dos métodos):
  * ``ctx = carregar_contexto(cfg)``; ``dev = carregar_particao("dev", cfg)``, em ordem cronológica.
  * Parâmetros em ``cfg["selecao"]`` (métodos, k_top), definidos antes de rodar.

Contrato de saída (consumido por ``src/votacao.py``, feature F4):
  * ``{método: Series(score por feature)}`` com os nomes DEPOIS do pré-processador e o mesmo índice nos
    três métodos. Maior score = melhor; todo score é >= 0. Empates são possíveis (a F4 define o desempate).

Sem vazamento: ``calcular_rankings_por_fold`` reajusta o pré-processador e os três métodos usando só o
treino de cada fold do ``criar_cv``. A validação do último fold não influencia nada, e o conjunto de
teste nem chega a este módulo.

Os nomes após o one-hot variam entre folds, porque as categorias acima de ``ohe_min_frequencia`` mudam
com o tempo. O índice final é a união dos folds; a feature ausente em um fold conta 0 na média dele, e
``tabela_rankings`` informa em quantos folds cada uma existiu.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from src.avaliacao import criar_cv
from src.config import RAIZ, resolver
from src.preprocessamento import construir_preprocessador

DIR_EVIDENCIAS = "evidencias/selecao"


def _score_selectkbest(X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> np.ndarray:
    """Informação mútua entre cada feature e o alvo (univariado)."""
    if params["score_func"] != "mutual_info_classif":
        raise ValueError(f"score_func não suportada: {params['score_func']}")
    # Colunas 0/1 (one-hot e binárias) são discretas; o padrão do sklearn as trataria como contínuas.
    discretas = np.isin(X.to_numpy(), [0.0, 1.0]).all(axis=0)
    score_func = partial(mutual_info_classif, discrete_features=discretas, random_state=seed)
    return SelectKBest(score_func, k="all").fit(X, y).scores_


def _score_random_forest(X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> np.ndarray:
    """Importância por redução de impureza; captura interações e não linearidades."""
    return RandomForestClassifier(**params, random_state=seed).fit(X, y).feature_importances_


def _score_rfe(X: pd.DataFrame, y: pd.Series, params: dict, seed: int, k_top: int) -> np.ndarray:
    """Eliminação recursiva até sobrarem ``k_top`` features.

    O ``ranking_`` do RFE é 1 para as mantidas e cresce com a eliminação. Vira ``(R - r + 1) / R``, em que
    R é o maior ranking: as ``k_top`` mantidas valem 1 e, como o ``step`` remove várias de uma vez, as
    eliminadas juntas empatam.
    """
    if params["estimador"] != "logistic_regression":
        raise ValueError(f"Estimador de RFE não suportado: {params['estimador']}")
    estimador = LogisticRegression(**params.get("params", {}), random_state=seed)
    ranking = RFE(estimador, n_features_to_select=min(k_top, X.shape[1]), step=params["step"]).fit(X, y).ranking_
    return (ranking.max() - ranking + 1) / ranking.max()


def calcular_rankings(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, pd.Series]:
    """Roda SelectKBest, importância do Random Forest e RFE sobre o mesmo ``X`` já pré-processado.

    Quem chama garante que ``X`` e ``y`` são só dados de treino; para o ciclo completo use
    ``calcular_rankings_por_fold``.
    """
    if not X.index.equals(y.index):
        raise ValueError("X e y precisam ter o mesmo índice")
    selecao, seed = cfg["selecao"], cfg["projeto"]["seed"]
    metodos = selecao["metodos"]
    scores = {
        "selectkbest": _score_selectkbest(X, y, metodos["selectkbest"], seed),
        "random_forest": _score_random_forest(X, y, metodos["random_forest"], seed),
        "rfe": _score_rfe(X, y, metodos["rfe"], seed, selecao["k_top"]),
    }
    return {metodo: pd.Series(valores, index=X.columns, name=metodo) for metodo, valores in scores.items()}


def calcular_rankings_por_fold(
    X_bruto: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Calcula os três rankings no treino de cada fold e faz a média dos scores entre os folds.

    ``X_bruto`` tem as colunas ANTES do pré-processador e deve estar em ordem cronológica.
    Devolve ``(rankings, por_fold)``: o contrato da F4 e a tabela longa ``fold, metodo, feature, score``.
    """
    partes = []
    n_folds = 0
    for n_folds, (treino, _) in enumerate(criar_cv(cfg).split(X_bruto), start=1):
        X_treino = construir_preprocessador(cfg).fit_transform(X_bruto.iloc[treino])
        for metodo, score in calcular_rankings(X_treino, y.iloc[treino], cfg).items():
            partes.append(pd.DataFrame({"fold": n_folds, "metodo": metodo, "feature": score.index, "score": score.to_numpy()}))
    por_fold = pd.concat(partes, ignore_index=True)
    # Soma / n_folds, e não média simples: fold em que a feature não existe conta como zero.
    medias = (por_fold.groupby(["metodo", "feature"])["score"].sum() / n_folds).unstack("metodo")
    rankings = {metodo: medias[metodo].rename(metodo) for metodo in cfg["selecao"]["metodos"]}
    return rankings, por_fold


def tabela_rankings(rankings: dict[str, pd.Series], por_fold: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por feature: score e posição (1 = melhor) em cada método e folds em que existiu."""
    colunas = {}
    for metodo, score in rankings.items():
        colunas[f"score_{metodo}"] = score
        colunas[f"posicao_{metodo}"] = score.rank(ascending=False, method="min").astype(int)
    tabela = pd.DataFrame(colunas)
    tabela["folds_presente"] = por_fold.groupby("feature")["fold"].nunique()
    tabela.index.name = "feature"
    return tabela


def salvar_rankings(tabela: pd.DataFrame, por_fold: pd.DataFrame, ctx: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    """Grava as tabelas em ``evidencias/selecao/``, identificadas pelo ``cycle_id`` e ``data_version``."""
    destino = resolver(DIR_EVIDENCIAS, raiz)
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "rankings": destino / f"rankings_{ctx['cycle_id']}.csv",
        "por_fold": destino / f"rankings_por_fold_{ctx['cycle_id']}.csv",
    }
    identificacao = {"cycle_id": ctx["cycle_id"], "data_version": ctx["data_version"]}
    tabela.reset_index().assign(**identificacao).to_csv(caminhos["rankings"], index=False, float_format="%.6f")
    por_fold.assign(**identificacao).to_csv(caminhos["por_fold"], index=False, float_format="%.6f")
    return caminhos
