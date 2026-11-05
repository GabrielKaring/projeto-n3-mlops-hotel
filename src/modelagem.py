"""Feature F5 (critério 5): dois algoritmos disputam o modelo preditivo.

Dono: ver ``config/equipe.yaml``. STATUS: a implementar. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Contrato de entrada (notebook 03):
  * ``features_eleitas.json`` gerado pela feature F4 (notebook 02) (mesmo cycle_id e data_version).
  * ``dev`` de ``carregar_particao("dev", cfg)``; modelos A e B em ``cfg["modelos"]``.

Regras do enunciado:
  * Modelo A e Modelo B usam as MESMAS features eleitas, a mesma partição e a mesma métrica.
  * Avaliar com ``src.avaliacao.dividir_cv`` e ``metricas_classificacao`` (protocolo único).
  * Cada modelo é um Pipeline completo (pré-processador + seleção das features eleitas + algoritmo),
    para que a serialização leve junto tudo que a previsão precisa.
  * Registrar hiperparâmetros, seed, métricas, features e artefatos dos dois no MLflow, com o mesmo cycle_id.

Contrato de saída (para o notebook 04):
  * ``models/candidatos/<cycle_id>/modelo_A.joblib`` e ``modelo_B.joblib``
  * ``data/processed/resultados_competicao.json`` com as métricas de A e B no mesmo protocolo.
"""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def construir_pipeline(nome_modelo: str, features_eleitas: list[str], cfg: dict) -> Pipeline:
    """Monta o Pipeline do modelo ``'A'`` ou ``'B'`` a partir de ``cfg['modelos']``."""
    raise NotImplementedError("TODO(F5): pré-processador + seletor das eleitas + algoritmo")


def treinar_e_avaliar(pipeline: Pipeline, dev: pd.DataFrame, cfg: dict) -> dict:
    """Valida com CV cronológico, treina no dev inteiro e devolve métricas no formato de ``metricas_classificacao``."""
    raise NotImplementedError("TODO(F5): treino e avaliação sob o protocolo comum")
