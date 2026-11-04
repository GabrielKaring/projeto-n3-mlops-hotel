"""Leitura da configuração e do contexto do ciclo compartilhado entre notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
CONFIG_PADRAO = RAIZ / "config" / "config.yaml"
NOME_CONTEXTO = "ciclo_atual.json"


def carregar_config(caminho: str | Path | None = None) -> dict:
    with open(caminho or CONFIG_PADRAO, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver(cfg_caminho: str | Path, raiz: Path = RAIZ) -> Path:
    """Resolve um caminho relativo da config a partir da raiz do projeto."""
    p = Path(cfg_caminho)
    return p if p.is_absolute() else raiz / p


def caminho_contexto(cfg: dict, raiz: Path = RAIZ) -> Path:
    return resolver(cfg["dados"]["dir_processed"], raiz) / NOME_CONTEXTO


def salvar_contexto(ctx: dict, cfg: dict, raiz: Path = RAIZ) -> Path:
    """Grava o contexto do ciclo (cycle_id, data_version, artefatos).

    É o contrato entre os notebooks: o 01 escreve, os 02 a 05 apenas leem.
    """
    destino = caminho_contexto(cfg, raiz)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(ctx, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return destino


def carregar_contexto(cfg: dict, raiz: Path = RAIZ) -> dict:
    destino = caminho_contexto(cfg, raiz)
    if not destino.exists():
        raise FileNotFoundError(
            f"Contexto do ciclo não encontrado em {destino}. Execute o notebook 01_etl antes."
        )
    return json.loads(destino.read_text(encoding="utf-8"))
