"""F10: PSI, escolha da referência, drift do alvo, alerta e evidências em dados sintéticos."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import drift, ingestao
from src.config import salvar_contexto
from src.preprocessamento import limpar_e_engenhar, lista_features
from tests.helpers import gerar_reservas


def _com_lote(partes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta lotes brutos no formato de ``consolidar_lotes`` (com ``lote_id`` e ``linha_origem``)."""
    saida = []
    for lote_id, p in partes.items():
        p = p.copy()
        p.insert(0, "linha_origem", range(len(p)))
        p.insert(0, "lote_id", lote_id)
        saida.append(p)
    return pd.concat(saida, ignore_index=True)


def _deslocar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Drift forte: antecedência 300 dias maior e todas as reservas canceladas."""
    b = bruto.copy()
    b["lead_time"] = b["lead_time"] + 300
    b["is_canceled"] = 1
    return b


def _projeto(raiz: Path, lotes: dict[str, pd.DataFrame]) -> Path:
    """Monta ``data/raw`` com os lotes e o manifesto dentro de ``raiz``."""
    raw = raiz / "data" / "raw"
    raw.mkdir(parents=True)
    for lote_id, df in lotes.items():
        df.to_csv(raw / f"{lote_id}.csv.gz", index=False)
    ingestao.registrar_lotes_novos(raw, raw / "manifest.json", agora=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return raiz


# --------------------------------------------------------------------------- regra na config


def test_regra_de_drift_esta_na_config(cfg):
    d = cfg["drift"]
    assert 0 < d["psi"]["atencao"] < d["psi"]["alerta"]
    assert 0 < d["alvo"]["atencao_pp"] < d["alvo"]["alerta_pp"]
    assert set(d["ignorar"]) <= set(lista_features(cfg))
    assert set(drift.colunas_monitoradas(cfg)).isdisjoint(d["ignorar"])


# --------------------------------------------------------------------------- PSI


def test_psi_e_zero_para_amostras_iguais():
    s = pd.Series(np.random.default_rng(0).normal(100, 30, 2000))
    assert drift.psi_numerica(s, s) == pytest.approx(0.0, abs=1e-12)
    c = pd.Series(list("aabbbc") * 100)
    assert drift.psi_categorica(c, c) == pytest.approx(0.0, abs=1e-12)


def test_psi_numerica_tolera_ruido_e_detecta_deslocamento():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(100, 30, 5000))
    mesma = pd.Series(rng.normal(100, 30, 5000))
    deslocada = pd.Series(rng.normal(160, 30, 5000))
    assert drift.psi_numerica(ref, mesma) < 0.10
    assert drift.psi_numerica(ref, deslocada) >= 0.25


def test_psi_categorica_agrupa_raras_e_detecta_categoria_nova_em_massa():
    ref = pd.Series(["a"] * 600 + ["b"] * 395 + ["x"] * 5)      # x tem 0,5%: rara
    atual = pd.Series(["a"] * 600 + ["b"] * 395 + ["y"] * 5)    # y é nova, e também rara
    assert drift.psi_categorica(ref, atual, min_frequencia=0.01) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(drift.psi_categorica(ref, atual))       # sem agrupar: finita mesmo com categoria ausente
    massa_nova = pd.Series(["a"] * 300 + ["b"] * 200 + ["y"] * 500)
    assert drift.psi_categorica(ref, massa_nova, min_frequencia=0.01) >= 0.25


@pytest.mark.parametrize(
    "valor,esperado",
    [(0.0, "ok"), (0.099, "ok"), (0.10, "atencao"), (0.249, "atencao"), (0.25, "alerta"), (3.0, "alerta")],
)
def test_classificar_respeita_os_limiares(valor, esperado):
    assert drift.classificar(valor, 0.10, 0.25) == esperado


# --------------------------------------------------------------------------- referência


def test_referencia_sazonal_usa_o_mesmo_mes_do_ano_anterior(cfg):
    lotes = [f"lote_2016-{m:02d}" for m in range(1, 13)] + ["lote_2017-01", "lote_2017-02"]
    assert drift.escolher_referencia(lotes, "lote_2017-02", cfg) == {"tipo": "sazonal", "lotes": ["lote_2016-02"]}


def test_sem_o_mesmo_mes_usa_a_janela_de_lotes_anteriores(cfg):
    c = copy.deepcopy(cfg)
    c["drift"]["janela_lotes"] = 2
    lotes = ["lote_2016-01", "lote_2016-02", "lote_2016-03", "lote_2016-04"]
    assert drift.escolher_referencia(lotes, "lote_2016-04", c) == {
        "tipo": "anteriores",
        "lotes": ["lote_2016-02", "lote_2016-03"],
    }


def test_referencia_nunca_usa_lotes_posteriores(cfg):
    lotes = ["lote_2016-04", "lote_2017-03", "lote_2017-04"]
    assert drift.escolher_referencia(lotes, "lote_2017-03", cfg) == {"tipo": "anteriores", "lotes": ["lote_2016-04"]}


def test_primeiro_lote_fica_sem_referencia(cfg):
    df, _ = limpar_e_engenhar(_com_lote({"lote_2016-01": gerar_reservas(600, 2016, 1)}), cfg)
    tabela, resumo = drift.monitorar_lote(df, "lote_2016-01", cfg)
    assert resumo["status"] == "sem_referencia" and tabela.empty


# --------------------------------------------------------------------------- avaliação de um lote


@pytest.fixture()
def bruto_sazonal() -> pd.DataFrame:
    return _com_lote({
        "lote_2016-04": gerar_reservas(3000, 2016, 4),
        "lote_2016-05": gerar_reservas(3000, 2016, 5),
        "lote_2017-04": gerar_reservas(3000, 2017, 4),
    })


def test_mesma_distribuicao_nao_gera_alerta(bruto_sazonal, cfg):
    df, _ = limpar_e_engenhar(bruto_sazonal, cfg)
    tabela, resumo = drift.monitorar_lote(df, "lote_2017-04", cfg)
    assert resumo["referencia"] == {"tipo": "sazonal", "lotes": ["lote_2016-04"]}
    assert resumo["n_features_alerta"] == 0
    assert resumo["status"] != "alerta"
    assert set(tabela["feature"]) == set(drift.colunas_monitoradas(cfg))


def test_lote_deslocado_gera_alerta_no_alvo_e_na_feature(bruto_sazonal, cfg):
    bruto = bruto_sazonal.copy()
    atual = bruto["lote_id"] == "lote_2017-04"
    bruto.loc[atual, "lead_time"] += 300
    bruto.loc[atual, "is_canceled"] = 1
    df, _ = limpar_e_engenhar(bruto, cfg)
    tabela, resumo = drift.monitorar_lote(df, "lote_2017-04", cfg)
    assert resumo["status"] == "alerta"
    assert resumo["alvo"]["status"] == "alerta" and resumo["alvo"]["diferenca_pp"] > 40
    assert "lead_time" in resumo["features_alerta"]
    assert tabela.iloc[0]["feature"] == "lead_time"


def test_lote_pequeno_nao_dispara_alerta(cfg):
    bruto = _com_lote({
        "lote_2016-04": gerar_reservas(3000, 2016, 4),
        "lote_2017-04": _deslocar(gerar_reservas(100, 2017, 4)),
    })
    df, _ = limpar_e_engenhar(bruto, cfg)
    _, resumo = drift.monitorar_lote(df, "lote_2017-04", cfg)
    assert resumo["status"] == "amostra_insuficiente"


# --------------------------------------------------------------------------- fluxo completo e evidências


def test_executar_avalia_o_lote_mais_recente_e_grava_evidencias(tmp_path, cfg, capsys):
    raiz = _projeto(tmp_path, {
        "lote_2016-04": gerar_reservas(800, 2016, 4),
        "lote_2016-05": gerar_reservas(800, 2016, 5),
        "lote_2017-04": _deslocar(gerar_reservas(800, 2017, 4)),
    })
    r = drift.executar(cfg, raiz=raiz, historico=True, ambiente={})

    assert r["resumo"]["lote_atual"] == "lote_2017-04"
    assert r["resumo"]["status"] == "alerta"
    assert r["ctx"]["origem_contexto"] == "avulso" and r["ctx"]["cycle_id"].endswith("-local")

    registro = json.loads(r["caminhos"]["json"].read_text(encoding="utf-8"))
    assert registro["cycle_id"] == r["ctx"]["cycle_id"]
    assert registro["data_version"].startswith("dv003-")
    assert registro["referencia"]["lotes"] == ["lote_2016-04"]
    for chave in ("csv", "png", "historico_csv", "historico_png"):
        assert r["caminhos"][chave].exists(), chave

    hist = pd.read_csv(r["caminhos"]["historico_csv"]).set_index("lote")
    assert hist.loc["lote_2016-04", "status"] == "sem_referencia"
    assert hist.loc["lote_2017-04", "status"] == "alerta"

    # Só lê dados: não cria nem altera o contexto do ciclo que os notebooks 02 a 05 usam.
    assert not (raiz / "data" / "processed" / "ciclo_atual.json").exists()
    assert "ALERTA DE DRIFT" in capsys.readouterr().out


def test_usa_o_cycle_id_do_notebook_01(tmp_path, cfg):
    raiz = _projeto(tmp_path, {"lote_2016-04": gerar_reservas(800, 2016, 4), "lote_2017-04": gerar_reservas(800, 2017, 4)})
    manifesto = ingestao.carregar_manifesto(raiz / "data" / "raw" / "manifest.json")
    salvar_contexto({"cycle_id": "20260101T000000Z-gha1-1", "data_version": ingestao.calcular_data_version(manifesto)}, cfg, raiz)
    r = drift.executar(cfg, raiz=raiz, ambiente={})
    assert r["ctx"]["cycle_id"] == "20260101T000000Z-gha1-1"
    assert r["caminhos"]["json"].name == "drift_20260101T000000Z-gha1-1.json"


def test_contexto_de_outra_versao_dos_dados_interrompe(tmp_path, cfg):
    raiz = _projeto(tmp_path, {"lote_2016-04": gerar_reservas(800, 2016, 4), "lote_2017-04": gerar_reservas(800, 2017, 4)})
    salvar_contexto({"cycle_id": "20260101T000000Z-local", "data_version": "dv001-antigo00"}, cfg, raiz)
    with pytest.raises(ValueError, match="notebook 01"):
        drift.executar(cfg, raiz=raiz, ambiente={})


def test_no_github_actions_gera_anotacao_e_resumo_do_job(tmp_path, capsys):
    resumo = {
        "lote_atual": "lote_2017-04",
        "referencia": {"tipo": "sazonal", "lotes": ["lote_2016-04"]},
        "status": "alerta",
        "alvo": {"taxa_referencia": 0.37, "taxa_atual": 0.43, "diferenca_pp": 6.0, "status": "alerta"},
        "features_alerta": ["lead_time"],
    }
    tabela = pd.DataFrame([{"feature": "lead_time", "tipo": "numerica", "psi": 0.31, "status": "alerta"}])
    arquivo_resumo = tmp_path / "summary.md"
    ambiente = {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(arquivo_resumo)}

    drift.emitir_alerta(resumo, tabela, {"cycle_id": "c1", "data_version": "dv001-x"}, ambiente)

    saida = capsys.readouterr().out
    assert saida.startswith("::warning title=Drift de dados::lote_2017-04")
    assert "%25" in saida and "+6.0 p.p." in saida
    texto = arquivo_resumo.read_text(encoding="utf-8")
    assert "**ALERTA**" in texto and "| lead_time | numerica | 0.310 | alerta |" in texto
