"""Fixtures compartilhadas pelos testes."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.config import carregar_config
from tests.helpers import gerar_reservas


@pytest.fixture(scope="session")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture()
def lotes_dir(tmp_path: Path) -> Path:
    """Diretório com três lotes mensais válidos (jan, fev, mar de 2016)."""
    d = tmp_path / "raw"
    d.mkdir()
    for mes in (1, 2, 3):
        gerar_reservas(200, 2016, mes).to_csv(d / f"lote_2016-{mes:02d}.csv.gz", index=False)
    return d


@pytest.fixture()
def df_bruto_multilote() -> pd.DataFrame:
    """DataFrame bruto de 6 meses já com ``lote_id`` e ``linha_origem`` (como sai de consolidar_lotes)."""
    partes = []
    for mes in range(1, 7):
        p = gerar_reservas(300, 2016, mes)
        p.insert(0, "linha_origem", range(len(p)))
        p.insert(0, "lote_id", f"lote_2016-{mes:02d}")
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


@pytest.fixture(scope="session")
def banco_modelo(cfg: dict, tmp_path_factory) -> Path:
    """Banco do MLflow vazio já migrado, criado uma vez só: criar do zero leva ~10 s por teste."""
    from src import rastreio

    c = copy.deepcopy(cfg)
    c["mlflow"]["dir_estado"] = str(tmp_path_factory.mktemp("modelo") / "estado")
    rastreio.configurar_mlflow(c)
    return Path(c["mlflow"]["dir_estado"]) / rastreio.NOME_BANCO
