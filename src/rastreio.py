"""Feature F6 (critério 6): MLflow e rastreabilidade.

Dono: ver ``config/equipe.yaml``. STATUS: a implementar. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Papel: oferecer às demais features uma forma única de registrar um ciclo no MLflow, sem que F3, F4, F5
e F8 precisem conhecer detalhes do MLflow. Parâmetros em ``cfg["mlflow"]``.

Requisitos do enunciado:
  * Todo run carrega ``cycle_id`` e ``data_version`` (tags ou parâmetros) e o ``run_id`` volta ao chamador.
  * Registrar parâmetros, métricas, features eleitas e artefatos (modelo, tabela de votos).
  * Dois ciclos diferentes precisam ser recuperáveis depois.
  * O runner do GitHub Actions é efêmero: definir onde o armazenamento do MLflow persiste entre ciclos
    (combinar com o dono da feature F7).
"""
from __future__ import annotations

from pathlib import Path


def iniciar_run(nome: str, ctx: dict, cfg: dict):
    """Abre um run no MLflow já com ``cycle_id`` e ``data_version`` como tags. Devolve o run ativo."""
    raise NotImplementedError("TODO(F6): configurar tracking URI/experimento e abrir o run com as tags do ciclo")


def registrar_artefatos(caminhos: list[Path], cfg: dict) -> None:
    """Anexa arquivos (tabela de votos, modelo serializado, métricas) ao run ativo."""
    raise NotImplementedError("TODO(F6): log_artifact dos arquivos do ciclo")
