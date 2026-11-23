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


def test_feature_dividida_conta_os_dois_donos():
    equipe = {
        "integrantes": [{"nome": "Ana"}, {"nome": "Beto"}, {"nome": "Caio"}],
        "features": {"F1": {"titulo": "t1", "dono": ["Ana", "Beto"]}},
    }
    commits = [
        {"hash": "a", "autor": "Ana", "data": "d", "assunto": "[F1] x", "features": ["F1"]},
        {"hash": "b", "autor": "Beto", "data": "d", "assunto": "[F1] y", "features": ["F1"]},
        {"hash": "c", "autor": "Caio", "data": "d", "assunto": "[F1] z", "features": ["F1"]},
    ]
    texto = "\n".join(gh.secao_por_feature(commits, equipe))
    assert "| F1 | t1 | Ana, Beto | 2 | 1 | ✅ com commits do dono |" in texto
    assert "| Ana | F1 | 1 |" in texto and "| Beto | F1 | 1 |" in texto and "| Caio | — | 0 |" in texto


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


def test_commit_sem_tag_recebe_a_feature_pelo_hash():
    equipe = {
        "integrantes": [{"nome": "Ana"}],
        "features": {"F7": {"titulo": "t7", "dono": "Ana"}},
        "features_por_commit": {"abc1234": ["F7"]},
    }
    commits = [
        {"hash": "abc1234", "autor": "Ana", "data": "d", "assunto": "implementa o workflow", "features": []},
        {"hash": "def5678", "autor": "Ana", "data": "d", "assunto": "[docs] z", "features": []},
    ]
    gh.aplicar_features_por_commit(commits, equipe)
    texto = "\n".join(gh.secao_por_feature(commits, equipe))
    assert "| F7 | t7 | Ana | 1 | 0 | ✅ com commits do dono |" in texto
    assert "atribuída em `config/equipe.yaml`" in texto and "1 commit(s) sem tag de feature" in texto


def test_commits_atribuidos_existem_e_nao_tem_tag():
    """Cada hash de features_por_commit existe no git, é de um commit sem tag e aponta para features válidas."""
    import subprocess
    equipe = gh.carregar_equipe()
    for h, fs in (equipe.get("features_por_commit") or {}).items():
        assunto = subprocess.run(["git", "log", "-1", "--format=%s", str(h)], cwd=gh.RAIZ,
                                 capture_output=True, text=True)
        if assunto.returncode != 0:      # clone raso (ex.: checkout do Actions): não há como conferir o hash
            pytest.skip("histórico do git incompleto")
        assert gh.extrair_features(assunto.stdout) == [] and set(fs) <= set(equipe["features"])
