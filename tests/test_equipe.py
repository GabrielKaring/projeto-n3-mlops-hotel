"""Garante a divisão de trabalho: 10 features, cada uma com um dono, e 1 a 3 features por pessoa."""
from collections import Counter
from pathlib import Path

import pytest
import yaml

EQUIPE = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "equipe.yaml").read_text(encoding="utf-8"))
FEATURES = EQUIPE["features"]
NOMES = [i["nome"] for i in EQUIPE["integrantes"]]


def test_features_cobrem_os_dez_criterios():
    assert sorted(FEATURES, key=lambda f: int(f[1:])) == [f"F{n}" for n in range(1, 11)]
    for fid, f in FEATURES.items():
        assert f["criterio"] == int(fid[1:]), f"{fid} deve corresponder ao critério {fid[1:]}"
        assert f["titulo"] and f["artefatos"]


def test_nomes_dos_integrantes_sao_unicos():
    assert len(NOMES) == len(set(NOMES))


def test_toda_feature_tem_dono_conhecido():
    for fid, f in FEATURES.items():
        assert f["dono"] in NOMES, f"{fid}: dono '{f['dono']}' não está em integrantes"


@pytest.mark.parametrize("nome", NOMES)
def test_cada_integrante_tem_de_1_a_3_features(nome):
    qtd = Counter(f["dono"] for f in FEATURES.values())[nome]
    assert 1 <= qtd <= 3, f"{nome} tem {qtd} feature(s); o combinado é de 1 a 3"
