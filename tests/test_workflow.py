"""F7: estrutura do workflow único. A execução real (inclusive pelo schedule) só é comprovada no Actions."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import RAIZ

DIR_WORKFLOWS = RAIZ / ".github" / "workflows"
WORKFLOW = DIR_WORKFLOWS / "pipeline_mlops.yml"
SCRIPT = RAIZ / "scripts" / "executar_notebook.sh"
ORDEM = ["01_etl", "02_votacao_features", "03_treino_competicao", "04_selecao_promocao", "05_previsoes"]


@pytest.fixture(scope="module")
def wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def passos(wf) -> list[dict]:
    return wf["jobs"]["pipeline"]["steps"]


def _gatilhos(wf: dict) -> dict:
    return wf.get("on", wf.get(True))  # o YAML 1.1 lê a chave "on" como booleano


def _indice(passos: list[dict], trecho: str) -> int:
    for i, passo in enumerate(passos):
        if trecho in passo.get("run", ""):
            return i
    raise AssertionError(f"Nenhum passo executa: {trecho}")


def test_existe_um_unico_workflow():
    arquivos = sorted(p.name for p in DIR_WORKFLOWS.iterdir() if p.suffix in (".yml", ".yaml"))
    assert arquivos == ["pipeline_mlops.yml"]


def test_tem_schedule_e_workflow_dispatch(wf):
    gatilhos = _gatilhos(wf)
    assert gatilhos["schedule"] and all("cron" in g for g in gatilhos["schedule"])
    assert "workflow_dispatch" in gatilhos


def test_notebooks_rodam_em_ordem_e_cada_um_em_seu_passo(passos):
    indices = [_indice(passos, f"executar_notebook.sh {nb}") for nb in ORDEM]
    assert indices == sorted(indices) and len(set(indices)) == len(ORDEM)


def test_falha_de_notebook_interrompe_o_fluxo(passos):
    texto = WORKFLOW.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert "--allow-errors" not in texto
    for nb in ORDEM:
        passo = passos[_indice(passos, f"executar_notebook.sh {nb}")]
        assert "continue-on-error" not in passo and "if" not in passo


def test_script_executa_o_notebook_e_simula_falha():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "jupyter nbconvert --to notebook --execute" in script
    assert "FALHAR_EM" in script and "exit 1" in script


def test_estado_e_recuperado_antes_dos_notebooks_e_guardado_depois(passos):
    restaurar = next(i for i, p in enumerate(passos) if p.get("id") == "restaurar")
    liberar = _indice(passos, "src.ingestao --liberar 1")
    primeiro, ultimo = _indice(passos, "executar_notebook.sh 01_etl"), _indice(passos, "executar_notebook.sh 05_previsoes")
    guardar = next(i for i, p in enumerate(passos) if p.get("name") == "Guardar estado persistente")
    assert restaurar < liberar < primeiro < ultimo < guardar
    assert "steps.restaurar.outcome == 'success'" in passos[guardar]["if"]


def test_vencedor_vai_para_a_release_entre_o_04_e_o_05(passos):
    release = next(i for i, p in enumerate(passos) if p.get("id") == "release")
    assert _indice(passos, "executar_notebook.sh 04_selecao_promocao") < release < _indice(passos, "executar_notebook.sh 05_previsoes")
    assert "gh release upload" in passos[release]["run"] and "--clobber" not in passos[release]["run"]


def test_permissoes_e_concorrencia(wf):
    assert wf["permissions"]["contents"] == "write"
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_instala_com_as_versoes_fixas(passos):
    """O runner instala as versões testadas (constraints.txt), não a última release de cada biblioteca."""
    comando = passos[_indice(passos, "pip install")]["run"]
    assert "-r requirements.txt" in comando and "-c constraints.txt" in comando
    fixas = {l.split("==")[0].lower() for l in (RAIZ / "constraints.txt").read_text(encoding="utf-8").splitlines()
             if "==" in l and not l.startswith("#")}
    assert {"pandas", "numpy", "scikit-learn", "mlflow", "pyarrow", "joblib", "nbconvert", "pytest"} <= fixas
