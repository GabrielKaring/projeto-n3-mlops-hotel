"""F8: eleição do vencedor, regra de promoção, manifesto auditável e previsões no próximo lote."""
from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.modelagem import construir_pipeline
from src.preprocessamento import limpar_e_engenhar, lista_features
from src.promocao import (
    atualizar_manifesto,
    avaliar_no_teste,
    carregar_manifesto,
    carregar_vigente,
    conferir_previsoes_anteriores,
    decidir_promocao,
    eleger_vencedor,
    guardar_modelo,
    nome_arquivo_modelo,
    preparar_lote,
    prever,
    proximo_lote,
    salvar_previsoes,
    tabela_historico,
    validar_resultados,
)
from tests.helpers import gerar_reservas

# Regra de config.yaml: f1_macro, maximizar, melhoria mínima 0,005, piso 0,60.


def _resultados(media_a, media_b, desvio_a=0.01, desvio_b=0.02, metrica="f1_macro"):
    def item(media, desvio):
        return {"algoritmo": "x", "params": {}, "cv": {"media": {metrica: media}, "desvio": {metrica: desvio}}}
    return {"cycle_id": "c1", "data_version": "dv1", "metrica_principal": metrica,
            "modelos": {"A": item(media_a, desvio_a), "B": item(media_b, desvio_b)}}


def _cand(valor):
    return {"metricas_teste": {"f1_macro": valor}}


def _registro(cycle_id, decisao, arquivo="m.joblib", valor=0.7):
    return {
        "cycle_id": cycle_id, "data_version": f"dv-{cycle_id}", "decisao": decisao, "motivo": "teste",
        "regra": {"metrica": "f1_macro"}, "valor_vencedor": valor, "valor_vigente": None,
        "candidato": {"modelo": "B", "algoritmo": "random_forest", "params": {}, "features": ["f"], "arquivo": arquivo},
    }


# ------------------------------------------------------------------------------------- eleição A x B

def test_vence_a_maior_media_no_cv(cfg):
    assert eleger_vencedor(_resultados(0.74, 0.78), cfg) == "B"
    assert eleger_vencedor(_resultados(0.80, 0.78), cfg) == "A"


def test_empate_na_media_vence_o_menor_desvio(cfg):
    assert eleger_vencedor(_resultados(0.75, 0.75, desvio_a=0.03, desvio_b=0.01), cfg) == "B"
    assert eleger_vencedor(_resultados(0.75, 0.75, desvio_a=0.01, desvio_b=0.03), cfg) == "A"


def test_minimizar_inverte_a_eleicao(cfg):
    c = copy.deepcopy(cfg)
    c["promocao"]["direcao"] = "minimizar"
    assert eleger_vencedor(_resultados(0.74, 0.78), c) == "A"


def test_resultados_de_outro_ciclo_falham(cfg):
    res = _resultados(0.7, 0.8)
    with pytest.raises(ValueError, match="outro ciclo"):
        validar_resultados(res, {"cycle_id": "outro", "data_version": "dv1"}, cfg)
    with pytest.raises(ValueError, match="promoção usa"):
        validar_resultados(_resultados(0.7, 0.8, metrica="roc_auc"), {"cycle_id": "c1", "data_version": "dv1"}, cfg)
    validar_resultados(res, {"cycle_id": "c1", "data_version": "dv1"}, cfg)


# ------------------------------------------------------------------------------------- regra de promoção

def test_primeiro_ciclo_promove_se_passa_no_piso(cfg):
    assert decidir_promocao(_cand(0.65), None, cfg)["decisao"] == "promovido"


def test_primeiro_ciclo_abaixo_do_piso_nao_promove(cfg):
    d = decidir_promocao(_cand(0.55), None, cfg)
    assert d["decisao"] == "mantido" and "piso" in d["motivo"]


def test_melhora_suficiente_promove(cfg):
    d = decidir_promocao(_cand(0.710), _cand(0.700), cfg)
    assert d["decisao"] == "promovido"
    assert d["valor_vigente"] == pytest.approx(0.700)


def test_melhora_exatamente_no_limiar_promove(cfg):
    assert decidir_promocao(_cand(0.705), _cand(0.700), cfg)["decisao"] == "promovido"


def test_melhora_insuficiente_mantem(cfg):
    assert decidir_promocao(_cand(0.703), _cand(0.700), cfg)["decisao"] == "mantido"


def test_piora_mantem(cfg):
    assert decidir_promocao(_cand(0.690), _cand(0.700), cfg)["decisao"] == "mantido"


def test_abaixo_do_piso_mantem_mesmo_melhorando(cfg):
    d = decidir_promocao(_cand(0.59), _cand(0.50), cfg)
    assert d["decisao"] == "mantido" and "piso" in d["motivo"]


# ------------------------------------------------------------------------------------- manifesto e modelos

def test_manifesto_preserva_historico_e_so_troca_vigente_na_promocao(cfg, tmp_path):
    assert carregar_manifesto(cfg, tmp_path)["vigente"] is None
    atualizar_manifesto(_registro("c1", "promovido", "m1.joblib"), cfg, tmp_path)
    atualizar_manifesto(_registro("c2", "mantido", "m2.joblib"), cfg, tmp_path)
    m = carregar_manifesto(cfg, tmp_path)
    assert m["vigente"]["cycle_id"] == "c1" and m["vigente"]["arquivo"] == "m1.joblib"
    assert [h["cycle_id"] for h in m["historico"]] == ["c1", "c2"]

    atualizar_manifesto(_registro("c3", "promovido", "m3.joblib"), cfg, tmp_path)
    m = carregar_manifesto(cfg, tmp_path)
    assert m["vigente"]["cycle_id"] == "c3"
    assert len(m["historico"]) == 3
    assert list(tabela_historico(m)["decisao"]) == ["promovido", "mantido", "promovido"]


def test_ciclo_ja_decidido_nao_e_registrado_de_novo(cfg, tmp_path):
    atualizar_manifesto(_registro("c1", "promovido"), cfg, tmp_path)
    with pytest.raises(ValueError, match="já tem decisão"):
        atualizar_manifesto(_registro("c1", "mantido"), cfg, tmp_path)


def test_decisao_invalida_falha(cfg, tmp_path):
    with pytest.raises(ValueError):
        atualizar_manifesto(_registro("c1", "talvez"), cfg, tmp_path)


def test_vigente_sem_arquivo_falha_em_vez_de_promover(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_vigente({"vigente": {"arquivo": "nao_existe.joblib"}}, tmp_path)
    assert carregar_vigente({"vigente": None}, tmp_path) == (None, None)


def test_modelo_guardado_nao_e_sobrescrito_e_volta_igual(tmp_path):
    caminho = guardar_modelo({"pesos": [1, 2, 3]}, "c1", "B", tmp_path)
    assert caminho.name == nome_arquivo_modelo("c1", "B") == "modelo_c1_B.joblib"
    with pytest.raises(FileExistsError):
        guardar_modelo({"pesos": [9]}, "c1", "B", tmp_path)
    modelo, _ = carregar_vigente({"vigente": {"arquivo": caminho.name}}, tmp_path)
    assert modelo == {"pesos": [1, 2, 3]}


# ------------------------------------------------------------------------------------- previsões

@pytest.fixture(scope="module")
def cfg_rapido(cfg) -> dict:
    c = copy.deepcopy(cfg)
    c["modelos"]["B"]["params"].update(n_estimators=20, max_depth=4)
    return c


@pytest.fixture(scope="module")
def modelo(cfg_rapido):
    brutos = pd.concat([gerar_reservas(150, 2016, m, seed=m).assign(lote_id=f"2016-{m:02d}") for m in range(1, 5)], ignore_index=True)
    brutos["linha_origem"] = range(len(brutos))
    dev, _ = limpar_e_engenhar(brutos, cfg_rapido)
    pipe = construir_pipeline("A", ["lead_time", "adr", "deposit_type_Non Refund"], cfg_rapido)
    return pipe.fit(dev[lista_features(cfg_rapido)], dev[cfg_rapido["avaliacao"]["alvo"]])


def test_avaliar_no_teste_devolve_a_metrica_da_regra(modelo, cfg_rapido):
    brutos = gerar_reservas(120, 2016, 6).assign(lote_id="2016-06")
    brutos["linha_origem"] = range(len(brutos))
    teste, _ = limpar_e_engenhar(brutos, cfg_rapido)
    m = avaliar_no_teste(modelo, teste, cfg_rapido)
    assert 0 <= m[cfg_rapido["promocao"]["metrica"]] <= 1 and "roc_auc" in m


def test_sem_lote_em_incoming_nao_ha_o_que_prever(cfg, tmp_path):
    assert proximo_lote(cfg, tmp_path) is None
    (tmp_path / "data" / "incoming").mkdir(parents=True)
    assert proximo_lote(cfg, tmp_path) is None


def test_previsao_no_proximo_lote_e_conferencia_quando_ele_chega(modelo, cfg_rapido, tmp_path):
    incoming = tmp_path / "data" / "incoming"
    incoming.mkdir(parents=True)
    for mes in (7, 8):
        gerar_reservas(80, 2016, mes).to_csv(incoming / f"lote_2016-{mes:02d}.csv.gz", index=False)

    arquivo = proximo_lote(cfg_rapido, tmp_path)
    assert arquivo.name == "lote_2016-07.csv.gz"  # a mesma ordem em que src.ingestao libera
    lote, _ = preparar_lote(arquivo, cfg_rapido)
    previsoes = prever(modelo, lote, cfg_rapido)
    assert cfg_rapido["avaliacao"]["alvo"] not in previsoes.columns  # o rótulo não vai junto
    assert set(previsoes["previsao"]) <= {0, 1} and previsoes["prob_cancelamento"].between(0, 1).all()
    assert (previsoes["lote_id"] == "lote_2016-07").all()

    ctx = {"cycle_id": "c1", "data_version": "dv1"}
    salvar_previsoes(previsoes, ctx, {"cycle_id": "c0", "modelo": "A"}, tmp_path)

    # Antes de o lote entrar no pipeline, não há rótulo para conferir.
    assert conferir_previsoes_anteriores(cfg_rapido, tmp_path).empty

    # O lote chega no ciclo seguinte: passa para data/raw e entra no manifesto de lotes.
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    arquivo.rename(raw / arquivo.name)
    (raw / "manifest.json").write_text(json.dumps({"lotes": [{"lote_id": "lote_2016-07", "arquivo": arquivo.name}]}))
    conferencia = conferir_previsoes_anteriores(cfg_rapido, tmp_path)
    assert len(conferencia) == 1
    linha = conferencia.iloc[0]
    assert linha["lote_id"] == "lote_2016-07" and linha["vigente_cycle_id"] == "c0"
    assert linha["n"] == len(previsoes) and 0 <= linha["f1_macro"] <= 1
