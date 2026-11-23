"""Feature F6: registro no MLflow com as tags do ciclo, recuperação de ciclos e portabilidade do histórico."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src import rastreio
from src.votacao import eleger_features, votar


@pytest.fixture()
def cfg_mlflow(cfg: dict, tmp_path: Path, banco_modelo: Path) -> dict:
    """Config com o estado do MLflow numa pasta temporária (não toca no mlflow_estado/ do projeto)."""
    c = copy.deepcopy(cfg)
    estado = tmp_path / "estado"
    estado.mkdir()
    shutil.copy(banco_modelo, estado / rastreio.NOME_BANCO)  # a relocação ajusta o caminho dos artefatos
    c["mlflow"]["dir_estado"] = str(estado)
    return c


@pytest.fixture(autouse=True)
def fecha_runs_abertos():
    yield
    while mlflow.active_run():
        mlflow.end_run()


def contexto(cycle_id: str = "20261119T120000Z-local", data_version: str = "dv018-abcd1234") -> dict:
    return {"cycle_id": cycle_id, "data_version": data_version, "particao": {"dev_n": 1000}}


def test_run_tem_tags_do_ciclo_e_devolve_run_id(cfg_mlflow):
    ctx = contexto()
    with rastreio.iniciar_run("teste", ctx, cfg_mlflow) as run:
        rastreio.registrar_metricas({"f1_macro": 0.75, "ignorada": None})
    run_id = run.info.run_id

    salvo = MlflowClient().get_run(run_id)
    assert salvo.data.tags["cycle_id"] == ctx["cycle_id"]
    assert salvo.data.tags["data_version"] == ctx["data_version"]
    assert salvo.data.tags["etapa"] == "teste"
    assert salvo.data.tags["origem"] in {"local", "github-actions"}
    assert salvo.data.metrics == {"f1_macro": 0.75}
    assert salvo.info.status == "FINISHED"
    assert salvo.info.experiment_id == mlflow.get_experiment_by_name(cfg_mlflow["mlflow"]["experimento"]).experiment_id
    artefatos = {a.path for a in MlflowClient().list_artifacts(run_id, "contexto")}
    assert artefatos == {"contexto/ciclo_atual.json"}


@pytest.mark.parametrize("faltando", ["cycle_id", "data_version"])
def test_contexto_sem_identificacao_do_ciclo_e_recusado(cfg_mlflow, faltando):
    ctx = contexto()
    del ctx[faltando]
    with pytest.raises(ValueError, match=faltando):
        rastreio.iniciar_run("teste", ctx, cfg_mlflow)
    assert mlflow.active_run() is None


def test_registrar_fora_de_um_run_falha():
    with pytest.raises(RuntimeError, match="Nenhum run ativo"):
        rastreio.registrar_metricas({"x": 1})


def test_run_com_erro_fica_marcado_como_falho(cfg_mlflow):
    with pytest.raises(ZeroDivisionError):
        with rastreio.iniciar_run("quebra", contexto(), cfg_mlflow) as run:
            1 / 0
    assert MlflowClient().get_run(run.info.run_id).info.status == "FAILED"


def test_achatar_secoes_aninhadas():
    assert rastreio.achatar({"rf": {"n_estimators": 200, "params": {"max_iter": 1000}}, "k": 20}) == {
        "rf.n_estimators": "200",
        "rf.params.max_iter": "1000",
        "k": "20",
    }


def test_registrar_votacao_guarda_parametros_metricas_features_e_artefatos(cfg_mlflow, tmp_path):
    cfg = copy.deepcopy(cfg_mlflow)
    cfg["selecao"]["k_top"] = 3
    cfg["selecao"]["votacao"] = {"min_votos": 2, "desempate": "ranking_medio", "min_features": 2}
    nomes = [f"f{i}" for i in range(6)]
    rankings = {
        "selectkbest": pd.Series([6, 5, 4, 3, 2, 1], index=nomes, dtype=float),
        "random_forest": pd.Series([5, 6, 1, 4, 3, 2], index=nomes, dtype=float),
        "rfe": pd.Series([1, 6, 5, 4, 3, 2], index=nomes, dtype=float),
    }
    tabela_votos = votar(rankings, cfg)
    eleitas = eleger_features(tabela_votos, cfg)

    arquivos = {}
    for nome in ("rankings", "por_fold", "votos", "eleitas"):
        arquivos[nome] = tmp_path / f"{nome}.csv"
        arquivos[nome].write_text("a,b\n1,2\n", encoding="utf-8")
    run_id = rastreio.registrar_votacao(
        contexto(), cfg, tabela_votos, eleitas,
        {"rankings": arquivos["rankings"], "por_fold": arquivos["por_fold"]},
        {"votos": arquivos["votos"], "eleitas": arquivos["eleitas"]},
    )

    cliente = MlflowClient()
    dados = cliente.get_run(run_id).data
    assert dados.tags["etapa"] == "02_votacao"
    assert dados.params["selecao.k_top"] == "3"
    assert dados.params["selecao.votacao.min_votos"] == "2"
    assert dados.params["n_features"] == str(len(eleitas))
    assert dados.metrics["n_features_candidatas"] == len(nomes)
    assert dados.metrics["n_features_eleitas"] == len(eleitas)
    assert sum(dados.metrics[f"features_com_{v}_votos"] for v in range(4)) == len(nomes)

    assert {a.path for a in cliente.list_artifacts(run_id, "selecao")} == {"selecao/rankings.csv", "selecao/por_fold.csv"}
    assert {a.path for a in cliente.list_artifacts(run_id, "votacao")} == {"votacao/votos.csv", "votacao/eleitas.csv"}
    baixado = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="features.json", dst_path=str(tmp_path / "dl")))
    assert json.loads(baixado.read_text(encoding="utf-8"))["features"] == eleitas


def test_artefato_inexistente_falha_em_vez_de_sumir_em_silencio(cfg_mlflow, tmp_path):
    with rastreio.iniciar_run("teste", contexto(), cfg_mlflow):
        with pytest.raises(FileNotFoundError):
            rastreio.registrar_artefatos([tmp_path / "nao_existe.csv"])


def test_dois_ciclos_ficam_recuperaveis_e_filtraveis(cfg_mlflow):
    ciclo1 = contexto("20261119T120000Z-local", "dv018-aaaa1111")
    ciclo2 = contexto("20261126T120000Z-local", "dv019-bbbb2222")
    for ctx, f1 in ((ciclo1, 0.70), (ciclo2, 0.72)):
        with rastreio.iniciar_run("03_modelo_A", ctx, cfg_mlflow):
            rastreio.registrar_metricas({"f1_macro": f1})
        with rastreio.iniciar_run("04_promocao", ctx, cfg_mlflow):
            rastreio.registrar_decisao({"decisao": "promovido", "motivo": "teste"})

    historico = rastreio.historico_ciclos(cfg_mlflow)
    assert historico["cycle_id"].tolist() == [ciclo1["cycle_id"]] * 2 + [ciclo2["cycle_id"]] * 2
    assert set(historico["data_version"]) == {"dv018-aaaa1111", "dv019-bbbb2222"}
    assert historico["run_id"].is_unique
    assert set(historico["decisao"].dropna()) == {"promovido"}

    so_ciclo2 = rastreio.historico_ciclos(cfg_mlflow, cycle_id=ciclo2["cycle_id"])
    assert len(so_ciclo2) == 2
    assert so_ciclo2.loc[so_ciclo2["etapa"] == "03_modelo_A", "metrics.f1_macro"].item() == 0.72


def test_historico_vazio_tem_as_colunas_esperadas(cfg_mlflow):
    assert list(rastreio.historico_ciclos(cfg_mlflow).columns) == rastreio.COLUNAS_HISTORICO


def test_historico_continua_valido_quando_a_pasta_muda_de_lugar(cfg_mlflow, tmp_path):
    """Simula o runner do Actions: o estado é criado num caminho e restaurado em outro."""
    with rastreio.iniciar_run("02_votacao", contexto(), cfg_mlflow) as run:
        rastreio.registrar_features(["lead_time", "adr"])
    origem = Path(cfg_mlflow["mlflow"]["dir_estado"])

    destino = tmp_path / "outra_maquina" / "estado"
    shutil.copytree(origem, destino)
    shutil.rmtree(origem / rastreio.DIR_ARTEFATOS)  # os artefatos antigos deixam de existir
    cfg_novo = copy.deepcopy(cfg_mlflow)
    cfg_novo["mlflow"]["dir_estado"] = str(destino)

    historico = rastreio.historico_ciclos(cfg_novo)
    assert historico["run_id"].tolist() == [run.info.run_id]
    salvo = MlflowClient().get_run(run.info.run_id)
    assert Path(destino).resolve().as_posix() in salvo.info.artifact_uri
    baixado = Path(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="features.json", dst_path=str(tmp_path / "dl")))
    assert json.loads(baixado.read_text(encoding="utf-8"))["features"] == ["lead_time", "adr"]
