"""Etapas 3 e 4 do enunciado: três métodos de seleção de features e votação.

STATUS: a implementar (ver docs/divisao_de_tarefas.md, critérios 3 e 4).

Contrato de entrada (notebook 02):
  * ``ctx = carregar_contexto(cfg)``; ``dev = carregar_particao("dev", cfg)``
  * X = ``construir_preprocessador(cfg)`` ajustado SOMENTE no dev (ou dentro de cada fold do CV)
  * y = ``dev[cfg["avaliacao"]["alvo"]]``
  * Parâmetros em ``cfg["selecao"]`` (k_top, métodos, regra de votação), já definidos antes de rodar.

Contrato de saída (para o notebook 03):
  * ``data/processed/features_eleitas.json`` com: cycle_id, data_version, lista de features eleitas
    (nomes após o pré-processador), regra de votação usada e k_top.
  * ``evidencias/votacao/votos_<cycle_id>.csv`` com a tabela de votos e os rankings dos 3 métodos.
  * Registro no MLflow dos votos, do conjunto final, dos parâmetros e do cycle_id (critério 6).

Lembrete do enunciado: a seleção é ajustada apenas no treino e, com validação cruzada, dentro de cada fold.
"""
from __future__ import annotations

import pandas as pd


def calcular_rankings(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, pd.Series]:
    """Roda SelectKBest, importância do Random Forest e RFE.

    Retorna um dicionário ``{nome_do_metodo: Series(importância/score por feature)}``.
    """
    raise NotImplementedError("TODO(critério 3): implementar os três métodos de seleção")


def votar(rankings: dict[str, pd.Series], cfg: dict) -> pd.DataFrame:
    """Monta a tabela de votos: uma linha por feature, uma coluna de voto por método, total e ranking médio."""
    raise NotImplementedError("TODO(critério 4): aplicar o top-k de cada método e contar os votos")


def eleger_features(tabela_votos: pd.DataFrame, cfg: dict) -> list[str]:
    """Aplica a regra de ``cfg['selecao']['votacao']`` (mín. de votos, desempate, mín. de features)."""
    raise NotImplementedError("TODO(critério 4): eleger o conjunto final de features")
