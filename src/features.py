"""Feature F3 (critério 3): três métodos de seleção de features.

Dono: ver ``config/equipe.yaml``. STATUS: a implementar. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada (notebook 02, seção dos métodos):
  * ``ctx = carregar_contexto(cfg)``; ``dev = carregar_particao("dev", cfg)``
  * X = ``construir_preprocessador(cfg)`` ajustado SOMENTE no dev (ou dentro de cada fold do CV)
  * y = ``dev[cfg["avaliacao"]["alvo"]]``
  * Parâmetros em ``cfg["selecao"]`` (métodos, k_top), já definidos antes de rodar.

Contrato de saída (consumido por ``src/votacao.py``, feature F4):
  * ``calcular_rankings`` devolve ``{método: Series(score por feature)}`` com os nomes das features
    DEPOIS do pré-processador (mesmo índice nos três métodos).

Lembrete do enunciado: a seleção é ajustada apenas no treino e, com validação cruzada, dentro de cada fold.
"""
from __future__ import annotations

import pandas as pd


def calcular_rankings(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, pd.Series]:
    """Roda SelectKBest, importância do Random Forest e RFE.

    Retorna um dicionário ``{nome_do_metodo: Series(importância/score por feature)}``.
    """
    raise NotImplementedError("TODO(F3): implementar os três métodos de seleção")
