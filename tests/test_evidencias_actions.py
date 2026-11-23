"""Cópia das evidências do Actions para a main (scripts/copiar_evidencias_actions.py) e o que já está copiado."""
import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import RAIZ

_spec = importlib.util.spec_from_file_location("copiar_evidencias_actions", RAIZ / "scripts" / "copiar_evidencias_actions.py")
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def test_run_ids_vem_do_historico_de_decisoes_na_ordem():
    manifesto = {"historico": [
        {"cycle_id": "20261123T004826Z-gha45090760506-1"},
        {"cycle_id": "20261120T012057Z-local"},          # ciclo local: não tem execução no Actions
        {"cycle_id": "20261123T021937Z-gha45099413735-2"},
    ]}
    assert ce.run_ids_do_manifesto(manifesto) == ["45090760506", "45099413735"]
    assert ce.run_ids_do_manifesto({"historico": []}) == []


def test_arquivo_pertence_ao_run_pelo_cycle_id():
    assert ce.do_run("decisao_20261123T004826Z-gha45090760506-1", "45090760506")
    assert not ce.do_run("decisao_20261123T004826Z-gha45090760506-1", "4509076050")
    assert not ce.do_run("decisao_20261120T012057Z-local", "45090760506")


@pytest.mark.parametrize("sub,nome,run_id,esperado", [
    ("votacao", "votos_20261123T004826Z-gha1-1.csv", "1", True),      # votos de todo ciclo
    ("votacao", "votos_20261123T004826Z-gha1-1.csv", "2", False),     # arquivo de outro run
    ("drift", "drift_20261123T004826Z-gha1-1.json", "1", False),     # drift: só da última execução
    ("drift", "drift_20261123T101743Z-gha2-1.json", "2", True),
    ("drift", "drift_20261123T101743Z-gha2-1.csv", "2", False),      # mesma tabela do JSON
    ("drift", "historico_20261123T101743Z-gha2-1.png", "2", True),
    ("promocao", "decisao_20261123T101743Z-gha2-1.json", "2", False), # já está no manifesto
    ("promocao", "erros_20261123T101743Z-gha2-1.json", "2", True),
])
def test_so_vai_para_a_main_o_que_nao_se_repete(sub, nome, run_id, esperado):
    assert ce.vale_copiar(sub, nome, run_id, ultimo_run="2") is esperado


def test_previsoes_comprimidas_voltam_iguais(tmp_path):
    original = tmp_path / "previsoes_c1.csv"
    pd.DataFrame({"linha_origem": range(50), "prob_cancelamento": [0.25] * 50}).to_csv(original, index=False)
    conteudo = original.read_bytes()
    gz = ce.comprimir(original)
    assert gz.name == "previsoes_c1.csv.gz" and not original.exists()
    primeiro = gz.read_bytes()
    assert gzip.decompress(primeiro) == conteudo
    original.write_bytes(conteudo)
    assert ce.comprimir(original).read_bytes() == primeiro  # mtime=0: mesmo conteúdo, mesmo .gz (sem diff no git)


def test_limpar_destino_preserva_o_leiame(tmp_path, monkeypatch):
    destino = tmp_path / "actions"
    (destino / "drift").mkdir(parents=True)
    (destino / "drift" / "velho.csv").write_text("x")
    (destino / "LEIAME.md").write_text("fica")
    monkeypatch.setattr(ce, "DESTINO", destino)
    ce.limpar_destino()
    assert [p.name for p in destino.iterdir()] == ["LEIAME.md"]


def test_copia_fica_fora_das_pastas_que_os_notebooks_leem():
    """O 01 lê evidencias/etl/diagnostico_*.json e o 05 lê evidencias/previsoes/previsoes_*.csv (sem subpastas)."""
    assert ce.DESTINO == RAIZ / "evidencias" / "actions"
    assert {p.parent.name for p in ce.DESTINO.rglob("diagnostico_*.json")} <= {"etl"}
    # os diagnósticos locais de 04/11 foram para desenvolvimento_local/, fora do glob do notebook 01
    assert not list((RAIZ / "evidencias" / "etl").glob("diagnostico_20261104T*-local.json"))


def test_manifesto_copiado_tem_o_que_o_enunciado_pede():
    copia = RAIZ / "models" / "modelo_vigente_actions.json"
    manifesto = json.loads(copia.read_text(encoding="utf-8"))
    vigente = manifesto["vigente"]
    for chave in ("cycle_id", "data_version", "features", "metrica", "run_id_promocao", "arquivo"):
        assert vigente[chave], chave
    assert all(h["decisao"] in ("promovido", "mantido") for h in manifesto["historico"])
    for run_id in ce.run_ids_do_manifesto(manifesto):   # cada ciclo decidido tem a sua tabela de votos copiada
        assert list((ce.DESTINO / "votacao").glob(f"votos_*-gha{run_id}-*.csv")), run_id
