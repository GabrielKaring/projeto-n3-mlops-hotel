import json
import shutil
from datetime import datetime, timezone

import pytest

from src import ingestao
from tests.helpers import gerar_reservas


def _registrar(d):
    return ingestao.registrar_lotes_novos(d, d / "manifest.json", agora=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_registra_lotes_e_calcula_metadados(lotes_dir):
    r = _registrar(lotes_dir)
    assert [l["lote_id"] for l in r["novos"]] == ["lote_2016-01", "lote_2016-02", "lote_2016-03"]
    m = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    assert all(l["n_linhas"] == 200 and len(l["sha256"]) == 64 for l in m["lotes"])
    assert m["lotes"][0]["chegada_min"].startswith("2016-01")


def test_registro_e_idempotente(lotes_dir):
    _registrar(lotes_dir)
    antes = (lotes_dir / "manifest.json").read_text()
    r = _registrar(lotes_dir)
    assert r["novos"] == [] and r["duplicados_ignorados"] == []
    assert (lotes_dir / "manifest.json").read_text() == antes


def test_mesmo_conteudo_com_outro_nome_nao_duplica(lotes_dir):
    _registrar(lotes_dir)
    shutil.copy(lotes_dir / "lote_2016-02.csv.gz", lotes_dir / "reenvio_fevereiro.csv.gz")
    r = _registrar(lotes_dir)
    assert r["novos"] == [] and r["duplicados_ignorados"] == ["reenvio_fevereiro.csv.gz"]
    manifesto = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    assert len(ingestao.consolidar_lotes(lotes_dir, manifesto)) == 600


def test_data_version_deterministico_e_muda_com_lote_novo(lotes_dir):
    _registrar(lotes_dir)
    m1 = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    assert ingestao.calcular_data_version(m1) == ingestao.calcular_data_version(m1)
    assert ingestao.calcular_data_version(m1).startswith("dv003-")
    gerar_reservas(150, 2016, 4).to_csv(lotes_dir / "lote_2016-04.csv.gz", index=False)
    _registrar(lotes_dir)
    m2 = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    assert ingestao.calcular_data_version(m2).startswith("dv004-")
    assert ingestao.calcular_data_version(m1) != ingestao.calcular_data_version(m2)


def test_lote_alterado_apos_registro_falha(lotes_dir):
    _registrar(lotes_dir)
    gerar_reservas(10, 2016, 1, seed=99).to_csv(lotes_dir / "lote_2016-01.csv.gz", index=False)
    manifesto = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    with pytest.raises(ValueError, match="alterado"):
        ingestao.consolidar_lotes(lotes_dir, manifesto)


def test_lote_registrado_ausente_falha(lotes_dir):
    _registrar(lotes_dir)
    (lotes_dir / "lote_2016-03.csv.gz").unlink()
    manifesto = ingestao.carregar_manifesto(lotes_dir / "manifest.json")
    with pytest.raises(FileNotFoundError):
        ingestao.consolidar_lotes(lotes_dir, manifesto)


def test_lote_sem_colunas_esperadas_e_rejeitado(lotes_dir):
    gerar_reservas(50, 2016, 4).drop(columns=["adr"]).to_csv(lotes_dir / "lote_2016-04.csv.gz", index=False)
    with pytest.raises(ValueError, match="adr"):
        _registrar(lotes_dir)
    assert not (lotes_dir / "manifest.json").exists()


def test_consolidar_adiciona_rastreabilidade(lotes_dir):
    _registrar(lotes_dir)
    df = ingestao.consolidar_lotes(lotes_dir, ingestao.carregar_manifesto(lotes_dir / "manifest.json"))
    assert len(df) == 600
    assert set(df["lote_id"]) == {"lote_2016-01", "lote_2016-02", "lote_2016-03"}
    assert df.groupby("lote_id")["linha_origem"].max().tolist() == [199, 199, 199]


def test_liberar_proximo_lote_respeita_ordem(tmp_path):
    inc, raw = tmp_path / "incoming", tmp_path / "raw"
    inc.mkdir()
    for mes in (5, 4, 6):
        gerar_reservas(20, 2016, mes).to_csv(inc / f"lote_2016-{mes:02d}.csv.gz", index=False)
    assert ingestao.liberar_proximo_lote(inc, raw) == ["lote_2016-04.csv.gz"]
    assert ingestao.liberar_proximo_lote(inc, raw, quantidade=5) == ["lote_2016-05.csv.gz", "lote_2016-06.csv.gz"]
    assert ingestao.liberar_proximo_lote(inc, raw) == []


def test_cycle_id_local_e_github():
    agora = datetime(2026, 11, 4, 12, 30, 5, tzinfo=timezone.utc)
    assert ingestao.gerar_cycle_id(agora, ambiente={}) == "20261104T123005Z-local"
    env = {"GITHUB_RUN_ID": "987", "GITHUB_RUN_ATTEMPT": "2"}
    assert ingestao.gerar_cycle_id(agora, ambiente=env) == "20261104T123005Z-gha987-2"


def test_manifesto_e_json_valido(lotes_dir):
    _registrar(lotes_dir)
    m = json.loads((lotes_dir / "manifest.json").read_text())
    assert m["schema_version"] == 1 and len(m["lotes"]) == 3
