"""Testa a leitura das tags [Fn] usadas pelo histórico por feature."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("gerar_historico", Path(__file__).resolve().parents[1] / "scripts" / "gerar_historico.py")
gh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gh)


@pytest.mark.parametrize("assunto,esperado", [
    ("[F3] implementa SelectKBest", ["F3"]),
    ("[F3][F4] ajusta contrato entre seleção e votação", ["F3", "F4"]),
    ("[F6,F7] persiste o MLflow no artefato", ["F6", "F7"]),
    ("[F3][F3] repetido", ["F3"]),
    ("[docs] atualiza README", []),
    ("corrige typo [F3] no meio", []),
])
def test_extrair_features(assunto, esperado):
    assert gh.extrair_features(assunto) == esperado


def test_secao_por_feature_distingue_dono_de_apoio():
    equipe = {
        "integrantes": [{"nome": "Ana"}, {"nome": "Beto"}],
        "features": {"F1": {"titulo": "t1", "dono": "Ana"}, "F2": {"titulo": "t2", "dono": "Beto"}},
    }
    commits = [
        {"hash": "a", "autor": "Ana", "data": "d", "assunto": "[F1] x", "features": ["F1"]},
        {"hash": "b", "autor": "Beto", "data": "d", "assunto": "[F1] y", "features": ["F1"]},
        {"hash": "c", "autor": "Ana", "data": "d", "assunto": "[docs] z", "features": []},
    ]
    texto = "\n".join(gh.secao_por_feature(commits, equipe))
    assert "| F1 | t1 | Ana | 1 | 1 | ✅ com commits do dono |" in texto
    assert "| F2 | t2 | Beto | 0 | 0 | ⬜ sem commits do dono |" in texto
    assert "1 commit(s) sem tag de feature" in texto
