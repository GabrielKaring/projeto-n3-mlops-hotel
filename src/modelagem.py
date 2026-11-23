"""Feature F5 (critério 5): dois algoritmos disputam o modelo preditivo.

Dono: ver ``config/equipe.yaml``. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada (notebook 03):
  * ``features_eleitas.json`` gerado pela F4 (notebook 02), com o mesmo cycle_id e data_version.
  * ``dev`` de ``carregar_particao("dev", cfg)``; modelos A e B em ``cfg["modelos"]``.

Regras:
  * A e B usam as mesmas features eleitas, os mesmos folds de ``dividir_cv`` e a mesma métrica.
  * Cada modelo é um Pipeline completo (pré-processador + seleção das eleitas + algoritmo), então o
    arquivo salvo já sabe transformar uma reserva crua em previsão.
  * O teste não passa por aqui. Ele só aparece no notebook 04.

As eleitas têm os nomes de depois do one-hot. Uma categoria rara pode não virar coluna em algum treino;
nesse caso a coluna entra zerada (é o que o one-hot faria com a categoria ausente) e fica anotada em
``features_ausentes``, para A e B continuarem vendo exatamente as mesmas colunas.

Contrato de saída (para o notebook 04, F8):
  * ``models/candidatos/<cycle_id>/modelo_A.joblib`` e ``modelo_B.joblib``
  * ``data/processed/resultados_competicao.json``, no formato de ``montar_resultados``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.avaliacao import dividir_cv, metricas_classificacao
from src.config import RAIZ, resolver
from src.preprocessamento import construir_preprocessador, lista_features
from src.votacao import eleger_features, votar

ALGORITMOS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}
NOME_RESULTADOS = "resultados_competicao.json"
DIR_MODELOS = "models/candidatos"
DIR_EVIDENCIAS = "evidencias/competicao"
COMPRESSAO = 3


class SeletorFeatures(TransformerMixin, BaseEstimator):
    """Mantém só as features eleitas, na ordem dada. Coluna que não existir no ajuste entra zerada."""

    def __init__(self, features: list[str] | None = None):
        self.features = features

    def fit(self, X, y=None):
        if not self.features:
            raise ValueError("Nenhuma feature eleita foi informada")
        self.ausentes_ = [f for f in self.features if f not in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.reindex(columns=list(self.features), fill_value=0.0)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features, dtype=object)


def construir_pipeline(nome_modelo: str, features_eleitas: list[str], cfg: dict) -> Pipeline:
    """Monta o Pipeline do modelo ``'A'`` ou ``'B'`` a partir de ``cfg['modelos']``."""
    if nome_modelo not in cfg["modelos"]:
        raise ValueError(f"Modelo desconhecido: {nome_modelo}. Opções: {sorted(cfg['modelos'])}")
    spec = cfg["modelos"][nome_modelo]
    if spec["algoritmo"] not in ALGORITMOS:
        raise ValueError(f"Algoritmo não suportado: {spec['algoritmo']}")
    algoritmo = ALGORITMOS[spec["algoritmo"]](**spec.get("params", {}), random_state=cfg["projeto"]["seed"])
    return Pipeline([
        ("pre", construir_preprocessador(cfg)),
        ("selecao", SeletorFeatures(list(features_eleitas))),
        ("modelo", algoritmo),
    ])


def _resumir(por_fold: list[dict]) -> dict:
    tabela = pd.DataFrame(por_fold).drop(columns=["fold"])
    return {"media": tabela.mean().to_dict(), "desvio": tabela.std(ddof=1).to_dict()}


def avaliar_cv(pipeline: Pipeline, dev: pd.DataFrame, cfg: dict, features_por_fold: dict[int, list[str]] | None = None) -> dict:
    """Valida com os folds de ``dividir_cv``. Cada fold treina uma cópia nova do pipeline.

    Com ``features_por_fold`` (fold -> lista), cada fold usa o próprio conjunto em vez do eleito no ciclo.
    """
    alvo, tempo = cfg["avaliacao"]["alvo"], cfg["avaliacao"]["coluna_tempo"]
    X, y = dev[lista_features(cfg)], dev[alvo]
    por_fold, ausentes = [], {}
    for fold, (treino, valid) in enumerate(dividir_cv(dev[tempo], cfg), start=1):
        modelo = clone(pipeline)
        if features_por_fold is not None:
            modelo.set_params(selecao__features=features_por_fold[fold])
        modelo.fit(X.iloc[treino], y.iloc[treino])
        y_score = modelo.predict_proba(X.iloc[valid])[:, 1]
        metricas = metricas_classificacao(y.iloc[valid], (y_score >= 0.5).astype(int), y_score)
        por_fold.append({"fold": fold, **metricas})
        if modelo.named_steps["selecao"].ausentes_:
            ausentes[fold] = modelo.named_steps["selecao"].ausentes_
    return {**_resumir(por_fold), "por_fold": por_fold, "features_ausentes": ausentes}


def treinar_e_avaliar(pipeline: Pipeline, dev: pd.DataFrame, cfg: dict) -> dict:
    """Valida com CV cronológico e depois treina ``pipeline`` no dev inteiro (o objeto passado sai ajustado).

    Devolve o resultado de ``avaliar_cv``.
    """
    resultado = avaliar_cv(pipeline, dev, cfg)
    pipeline.fit(dev[lista_features(cfg)], dev[cfg["avaliacao"]["alvo"]])
    return resultado


def eleger_por_fold(por_fold: pd.DataFrame, cfg: dict) -> dict[int, list[str]]:
    """Refaz a votação da F4 com o ranking de cada fold isolado (tabela ``por_fold`` da F3).

    Serve para medir o otimismo do CV: o ranking do fold k vem só do treino do fold k, então a
    validação dele fica realmente de fora da escolha das features.
    """
    eleitas = {}
    for fold, parte in por_fold.groupby("fold"):
        scores = parte.pivot(index="feature", columns="metodo", values="score")
        rankings = {m: scores[m].rename(m) for m in cfg["selecao"]["metodos"]}
        eleitas[int(fold)] = eleger_features(votar(rankings, cfg), cfg)
    return eleitas


def montar_resultados(ctx: dict, cfg: dict, features_eleitas: list[str], avaliacoes: dict[str, dict],
                      sensibilidade: dict[str, dict] | None = None, caminhos_modelos: dict[str, Path] | None = None) -> dict:
    """Junta tudo no formato que o notebook 04 lê. ``avaliacoes`` é {'A': saída de treinar_e_avaliar, ...}."""
    metrica = cfg["avaliacao"]["metrica_principal"]
    modelos = {}
    for nome, avaliacao in avaliacoes.items():
        spec = cfg["modelos"][nome]
        item = {
            "algoritmo": spec["algoritmo"],
            "params": spec.get("params", {}),
            "seed": cfg["projeto"]["seed"],
            "cv": avaliacao,
            "metrica_principal": avaliacao["media"][metrica],
        }
        if sensibilidade and nome in sensibilidade:
            media_sens = sensibilidade[nome]["media"][metrica]
            item["sensibilidade_eleicao_por_fold"] = {
                **sensibilidade[nome],
                "diferenca_metrica_principal": avaliacao["media"][metrica] - media_sens,
            }
        if caminhos_modelos and nome in caminhos_modelos:
            item["caminho_modelo"] = str(caminhos_modelos[nome])
        modelos[nome] = item
    return {
        "cycle_id": ctx["cycle_id"],
        "data_version": ctx["data_version"],
        "metrica_principal": metrica,
        "direcao": cfg["avaliacao"].get("direcao", "maximizar"),
        "n_splits": cfg["avaliacao"]["cv"]["n_splits"],
        "features_eleitas": list(features_eleitas),
        "n_features": len(features_eleitas),
        "modelos": modelos,
    }


def salvar_modelos(pipelines: dict[str, Pipeline], ctx: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    """Serializa cada pipeline ajustado em ``models/candidatos/<cycle_id>/modelo_<nome>.joblib``."""
    destino = resolver(DIR_MODELOS, raiz) / ctx["cycle_id"]
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for nome, pipeline in pipelines.items():
        caminhos[nome] = destino / f"modelo_{nome}.joblib"
        # Comprimido (como em src.promocao): o B cai de ~50 MB para ~19 MB. O arquivo também vai para o MLflow,
        # cujo histórico o workflow commita na branch de estado a cada ciclo, e o GitHub avisa acima de 50 MB por arquivo.
        joblib.dump(pipeline, caminhos[nome], compress=COMPRESSAO)
    return caminhos


def salvar_resultados(resultados: dict, cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    """Grava o json para o notebook 04 e uma cópia com o cycle_id em ``evidencias/competicao/``."""
    texto = json.dumps(resultados, indent=2, ensure_ascii=False, default=str)
    caminho = resolver(cfg["dados"]["dir_processed"], raiz) / NOME_RESULTADOS
    evidencia = resolver(DIR_EVIDENCIAS, raiz) / f"resultados_{resultados['cycle_id']}.json"
    for p in (caminho, evidencia):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    return {"resultados": caminho, "evidencia": evidencia}
