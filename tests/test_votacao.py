"""F4: voto, desempate, piso de features e artefatos em casos pequenos."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.votacao import eleger_features, salvar_votacao, votar


@pytest.fixture()
def cfg_voto(cfg) -> dict:
    c = copy.deepcopy(cfg)
    c["selecao"]["k_top"] = 2
    c["selecao"]["votacao"] = {"min_votos": 2, "desempate": "ranking_medio", "min_features": 1}
    return c


@pytest.fixture()
def rankings() -> dict[str, pd.Series]:
    """Top-2: m1 = a, b; m2 = a, c; m3 = a, b. Logo a tem 3 votos, b tem 2, c tem 1 e d nenhum."""
    return {
        "m1": pd.Series({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}),
        "m2": pd.Series({"a": 0.9, "b": 0.3, "c": 0.8, "d": 0.1}),
        "m3": pd.Series({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}),
    }


def test_tabela_conta_os_votos_de_cada_metodo(rankings, cfg_voto):
    tabela = votar(rankings, cfg_voto)
    assert tabela["votos"].to_dict() == {"a": 3, "b": 2, "c": 1, "d": 0}
    assert tabela.loc["b", ["voto_m1", "voto_m2", "voto_m3"]].tolist() == [1, 0, 1]
    assert tabela.loc["b", "ranking_medio"] == pytest.approx((2 + 3 + 2) / 3)


def test_elege_quem_atinge_o_minimo_de_votos(rankings, cfg_voto):
    assert eleger_features(votar(rankings, cfg_voto), cfg_voto) == ["a", "b"]


def test_piso_completa_pelo_ranking_medio(rankings, cfg_voto):
    cfg_voto["selecao"]["votacao"]["min_features"] = 3
    assert eleger_features(votar(rankings, cfg_voto), cfg_voto) == ["a", "b", "c"]


def test_empate_de_votos_e_decidido_pelo_ranking_medio(cfg_voto):
    """x e y têm 1 voto cada, mas x tem melhor ranking médio: entra primeiro no piso."""
    cfg_voto["selecao"]["k_top"] = 1
    cfg_voto["selecao"]["votacao"].update(min_votos=3, min_features=1)
    empate = {
        "m1": pd.Series({"x": 0.9, "y": 0.8, "z": 0.1}),
        "m2": pd.Series({"x": 0.8, "y": 0.9, "z": 0.1}),
        "m3": pd.Series({"x": 0.9, "y": 0.1, "z": 0.8}),
    }
    tabela = votar(empate, cfg_voto)
    assert eleger_features(tabela, cfg_voto) == ["x"]  # nenhuma com 3 votos; o piso pega a melhor

    empate_total = {m: pd.Series({"y": 0.5, "x": 0.5}) for m in ("m1", "m2", "m3")}
    cfg_voto["selecao"]["votacao"].update(min_votos=1, min_features=1)
    assert eleger_features(votar(empate_total, cfg_voto), cfg_voto) == ["x", "y"]  # empate total: ordem pelo nome


def test_feature_empatada_na_fronteira_do_top_k_recebe_o_voto(cfg_voto):
    empate = {m: pd.Series({"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}) for m in ("m1", "m2", "m3")}
    assert votar(empate, cfg_voto)["votos"].to_dict() == {"a": 3, "b": 3, "c": 3, "d": 0}


@pytest.mark.parametrize("campo,valor,erro", [
    ("min_votos", 0, "min_votos"),
    ("min_votos", 4, "min_votos"),
    ("desempate", "sorteio", "Desempate"),
    ("min_features", 99, "min_features"),
])
def test_regra_invalida_falha(rankings, cfg_voto, campo, valor, erro):
    cfg_voto["selecao"]["votacao"][campo] = valor
    with pytest.raises(ValueError, match=erro):
        eleger_features(votar(rankings, cfg_voto), cfg_voto)


def test_regra_oficial_esta_na_config(cfg):
    assert cfg["selecao"]["votacao"] == {"min_votos": 2, "desempate": "ranking_medio", "min_features": 10}


def test_salvar_grava_os_votos_e_o_contrato_da_f5(rankings, cfg_voto, tmp_path: Path):
    tabela = votar(rankings, cfg_voto)
    eleitas = eleger_features(tabela, cfg_voto)
    ctx = {"cycle_id": "ciclo-x", "data_version": "dv001-abc"}
    caminhos = salvar_votacao(tabela, eleitas, ctx, cfg_voto, tmp_path)

    salvo = pd.read_csv(caminhos["votos"], index_col="feature")
    assert caminhos["votos"].name == "votos_ciclo-x.csv"
    assert salvo["eleita"].to_dict() == {"a": 1, "b": 1, "c": 0, "d": 0}
    assert set(salvo["cycle_id"]) == {"ciclo-x"} and set(salvo["data_version"]) == {"dv001-abc"}

    contrato = json.loads(caminhos["eleitas"].read_text(encoding="utf-8"))
    assert contrato["features_eleitas"] == ["a", "b"] and contrato["n_features"] == 2
    assert contrato["cycle_id"] == "ciclo-x" and contrato["data_version"] == "dv001-abc"
    assert contrato["k_top"] == 2 and contrato["regra_votacao"]["min_votos"] == 2
