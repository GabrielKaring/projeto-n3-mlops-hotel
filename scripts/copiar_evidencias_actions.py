"""Traz para a main uma cópia das evidências produzidas pelo GitHub Actions (F9). Só lê o GitHub.

O workflow guarda o estado na branch ``estado-pipeline`` e publica as evidências de cada execução no
artefato ``evidencias-<run_id>``. Nada disso fica na ``main``, que é onde o professor procura. Este script
copia, sem alterar nada no GitHub:

  * ``models/modelo_vigente.json`` da ``estado-pipeline``  ->  ``models/modelo_vigente_actions.json``;
  * da ``estado-pipeline``: as previsões de cada ciclo (comprimidas) e o diagnóstico do ETL da última execução;
  * do artefato de cada execução que decidiu um ciclo: a tabela de votos; da última execução, também o drift
    (JSON, gráfico e histórico de todos os lotes) e a análise de erros;
  * ``evidencias/mlflow/historico_runs.csv`` da última execução (todos os runs, com as métricas de A e B).

Para não inchar a main, fica de fora o que se repete (``vale_copiar``): as decisões já estão no histórico do
manifesto, as métricas da competição no ``historico_runs.csv`` e o drift de cada lote no histórico de drift.

Tudo vai para ``evidencias/actions/`` (fora das pastas que os notebooks leem, para não misturar com os
ciclos locais), refeita do zero a cada execução (o ``LEIAME.md`` fica). Uso, com o ``gh`` autenticado:

    python scripts/copiar_evidencias_actions.py
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BRANCH_ESTADO = "estado-pipeline"
DESTINO = RAIZ / "evidencias" / "actions"
MANIFESTO_COPIA = RAIZ / "models" / "modelo_vigente_actions.json"
DO_ESTADO = ("evidencias/etl/", "evidencias/previsoes/")
DO_ARTEFATO = ("promocao", "drift", "votacao")
_RUN = re.compile(r"-gha(\d+)-\d+$")


def run_ids_do_manifesto(manifesto: dict) -> list[str]:
    """IDs das execuções do Actions que decidiram um ciclo, na ordem do histórico."""
    ids = []
    for h in manifesto.get("historico", []):
        m = _RUN.search(h["cycle_id"])
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def do_run(nome: str, run_id: str) -> bool:
    """O arquivo é deste run? (``..._<timestamp>-gha<run_id>-<tentativa>.<ext>``)"""
    return f"-gha{run_id}-" in nome


def vale_copiar(sub: str, nome: str, run_id: str, ultimo_run: str) -> bool:
    """O arquivo do artefato vai para a main? Só os deste run, sem o que se repete entre arquivos e execuções."""
    if not do_run(Path(nome).stem, run_id):
        return False
    if sub == "votacao":
        return True                             # a tabela de votos de cada ciclo só existe aqui
    if sub == "drift" and nome.startswith("drift_") and nome.endswith(".csv"):
        return False                            # mesma tabela do drift_<cycle_id>.json
    if sub == "promocao" and nome.startswith("decisao_"):
        return False                            # já está no histórico do manifesto copiado
    return run_id == ultimo_run                 # drift (cumulativo) e análise de erros: só da última execução


def comprimir(arquivo: Path) -> Path:
    """Troca ``x.csv`` por ``x.csv.gz``. ``mtime=0`` deixa o .gz igual a cada cópia, sem diff à toa no git."""
    destino = arquivo.with_name(arquivo.name + ".gz")
    with open(destino, "wb") as bruto, gzip.GzipFile(filename="", mode="wb", fileobj=bruto, mtime=0) as gz:
        gz.write(arquivo.read_bytes())
    arquivo.unlink()
    return destino


def limpar_destino() -> None:
    """Apaga a cópia anterior (menos o ``LEIAME.md``), para não sobrar arquivo que o script não traz mais."""
    if DESTINO.is_dir():
        for item in DESTINO.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.name != "LEIAME.md":
                item.unlink()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout


def copiar_estado() -> dict:
    _git("fetch", "--quiet", "origin", BRANCH_ESTADO)
    ref = f"origin/{BRANCH_ESTADO}"
    manifesto = json.loads(_git("show", f"{ref}:models/modelo_vigente.json"))
    MANIFESTO_COPIA.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ultimo = run_ids_do_manifesto(manifesto)[-1]
    for caminho in _git("ls-tree", "-r", "--name-only", ref).splitlines():
        if caminho.startswith("evidencias/etl/") and not do_run(caminho, ultimo):
            continue                            # diagnóstico do ETL: só o da última execução
        if caminho.startswith(DO_ESTADO) and "-gha" in caminho:
            destino = DESTINO / Path(caminho).relative_to("evidencias")
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(subprocess.run(["git", "show", f"{ref}:{caminho}"], cwd=RAIZ, capture_output=True, check=True).stdout)
    return manifesto


def copiar_artefatos(run_ids: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for run_id in run_ids:
            pasta = Path(tmp) / run_id
            subprocess.run(["gh", "run", "download", run_id, "-n", f"evidencias-{run_id}", "-D", str(pasta)], cwd=RAIZ, check=True)
            for sub in DO_ARTEFATO:
                for arquivo in sorted((pasta / "evidencias" / sub).glob("*")):
                    if arquivo.is_file() and vale_copiar(sub, arquivo.name, run_id, run_ids[-1]):
                        (DESTINO / sub).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(arquivo, DESTINO / sub / arquivo.name)
            historico = pasta / "evidencias" / "mlflow" / "historico_runs.csv"
            if historico.exists():   # o último run sobrescreve: fica o histórico mais completo
                (DESTINO / "mlflow").mkdir(parents=True, exist_ok=True)
                shutil.copy2(historico, DESTINO / "mlflow" / historico.name)


def main() -> int:
    limpar_destino()
    manifesto = copiar_estado()
    run_ids = run_ids_do_manifesto(manifesto)
    copiar_artefatos(run_ids)
    for arquivo in sorted((DESTINO / "previsoes").glob("previsoes_*.csv")):
        comprimir(arquivo)
    print("execuções copiadas:", ", ".join(run_ids))
    print("manifesto:", MANIFESTO_COPIA.relative_to(RAIZ), "| evidências:", DESTINO.relative_to(RAIZ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
