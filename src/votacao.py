"""Feature F4 (critério 4): votação e eleição do conjunto final de features.

Dono: ver ``config/equipe.yaml``. STATUS: a implementar. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada:
  * ``rankings`` produzidos por ``src.features.calcular_rankings`` (feature F3).
  * Regra de votação em ``cfg["selecao"]["votacao"]``, definida ANTES de ver resultados.

Contrato de saída (para o notebook 03, feature F5):
  * ``data/processed/features_eleitas.json`` com: cycle_id, data_version, features eleitas
    (nomes após o pré-processador), regra de votação usada e k_top.
  * ``evidencias/votacao/votos_<cycle_id>.csv`` com a tabela de votos e os rankings dos 3 métodos.
  * Registro no MLflow dos votos e do conjunto final (feature F6 fornece a função de registro).
"""
from __future__ import annotations

import pandas as pd


def votar(rankings: dict[str, pd.Series], cfg: dict) -> pd.DataFrame:
    """Monta a tabela de votos: uma linha por feature, uma coluna de voto por método, total e ranking médio."""
    raise NotImplementedError("TODO(F4): aplicar o top-k de cada método e contar os votos")


def eleger_features(tabela_votos: pd.DataFrame, cfg: dict) -> list[str]:
    """Aplica a regra de ``cfg['selecao']['votacao']`` (mín. de votos, desempate, mín. de features)."""
    raise NotImplementedError("TODO(F4): eleger o conjunto final de features")
