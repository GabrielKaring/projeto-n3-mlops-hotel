"""Demonstração de uso: ETL no modo inferência e previsão de reservas novas, sem rótulo, com o vigente."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import prever
from src.config import RAIZ
from src.modelagem import construir_pipeline
from src.preprocessamento import limpar_e_engenhar, lista_features
from src.promocao import LIMIAR_CLASSE, atualizar_manifesto, guardar_modelo
from src.schema import ALVO, COLUNAS_POS_RESERVA, COLUNAS_RESERVA
from tests.helpers import gerar_reservas


def _reservas_novas(n: int = 30) -> pd.DataFrame:
    """Reservas como chegam no ato da reserva: sem desfecho e sem as colunas posteriores."""
    return gerar_reservas(n, 2017, 9, seed=7).drop(columns=[ALVO, *COLUNAS_POS_RESERVA])


def _json_reserva(pasta: Path) -> Path:
    entrada = pasta / "reserva.json"
    entrada.write_text(json.dumps(_reservas_novas(1).iloc[0].to_dict(), default=float))
    return entrada


@pytest.fixture(scope="module")
def modelo(cfg):
    brutos = pd.concat([gerar_reservas(150, 2016, m, seed=m).assign(lote_id=f"2016-{m:02d}") for m in range(1, 5)], ignore_index=True)
    brutos["linha_origem"] = range(len(brutos))
    dev, _ = limpar_e_engenhar(brutos, cfg)
    pipe = construir_pipeline("A", ["lead_time", "adr", "deposit_type_Non Refund"], cfg)
    return pipe.fit(dev[lista_features(cfg)], dev[cfg["avaliacao"]["alvo"]])


# ------------------------------------------------------------------------------------- ETL no modo inferência

def test_validacao_do_treino_e_da_inferencia(cfg):
    with pytest.raises(ValueError, match="Colunas ausentes"):            # o treino continua exigindo tudo
        limpar_e_engenhar(_reservas_novas(), cfg)
    with pytest.raises(ValueError, match="lead_time"):                   # a inferência exige o que se sabe na reserva
        limpar_e_engenhar(_reservas_novas().drop(columns=["lead_time"]), cfg, inferencia=True)


def test_inferencia_aceita_reserva_sem_alvo_e_sem_campos_opcionais(cfg):
    novas = _reservas_novas().drop(columns=["country", "agent", "company", "children"])
    limpo, rel = limpar_e_engenhar(novas, cfg, inferencia=True)
    assert ALVO not in limpo.columns and len(limpo) == len(novas) - rel["regras_dominio_removidas"]
    assert not limpo[lista_features(cfg)].isna().any().any()
    assert (limpo["agent"] == "SEM_AGENTE").all() and (limpo["tem_empresa"] == 0).all()


def test_inferencia_gera_as_mesmas_features_do_treino(cfg):
    """O modo inferência não muda nenhuma regra: mesmas linhas, mesmos atributos derivados."""
    completas = gerar_reservas(40, 2017, 9, seed=3)
    treino, _ = limpar_e_engenhar(completas.assign(lote_id="x", linha_origem=range(40)), cfg)
    inferencia, _ = limpar_e_engenhar(completas.drop(columns=[ALVO, *COLUNAS_POS_RESERVA]).assign(lote_id="x", linha_origem=range(40)), cfg, inferencia=True)
    feats = lista_features(cfg)
    a = treino.sort_values("linha_origem").reset_index(drop=True)[feats]
    b = inferencia.sort_values("linha_origem").reset_index(drop=True)[feats]
    pd.testing.assert_frame_equal(a, b)


def test_inferencia_mantem_a_ordem_e_nao_remove_duplicatas(cfg):
    novas = _reservas_novas(5)
    repetidas = pd.concat([novas, novas.iloc[[0]]], ignore_index=True)
    limpo, rel = limpar_e_engenhar(repetidas, cfg, inferencia=True)
    assert rel["duplicatas_removidas"] == 0
    assert list(limpo["lead_time"]) == list(repetidas["lead_time"])



# ------------------------------------------------------------------------------------- prever_reserva

def test_prever_uma_reserva_devolve_probabilidade_e_rotulo(modelo, cfg):
    reserva = _reservas_novas(1).iloc[0].to_dict()
    r = prever.prever_reserva(reserva, modelo, {"cycle_id": "c1", "modelo": "A"}, cfg)
    assert len(r) == 1
    linha = r.iloc[0]
    assert 0 <= linha["prob_cancelamento"] <= 1 and linha["limiar"] == LIMIAR_CLASSE
    esperado = prever.RISCO_ALTO if linha["prob_cancelamento"] >= LIMIAR_CLASSE else prever.RISCO_BAIXO
    assert linha["risco"] == esperado and linha["previsao"] == int(linha["prob_cancelamento"] >= LIMIAR_CLASSE)
    assert linha["vigente_cycle_id"] == "c1"
    texto = prever.descrever(r)
    assert "chance de cancelamento" in texto and "ciclo c1" in texto and "Limiar: 50%" in texto


def test_rotulo_que_venha_junto_e_ignorado(modelo, cfg):
    reservas = gerar_reservas(20, 2017, 9, seed=5)
    a = prever.prever_reserva(reservas.assign(is_canceled=0), modelo, cfg=cfg)
    b = prever.prever_reserva(reservas.assign(is_canceled=1), modelo, cfg=cfg)
    pd.testing.assert_series_equal(a["prob_cancelamento"], b["prob_cancelamento"])


def test_varias_reservas_na_ordem_da_entrada(modelo, cfg):
    novas = _reservas_novas(8)
    r = prever.prever_reserva(novas.to_dict(orient="records"), modelo, cfg=cfg)
    assert list(r["reserva"]) == list(range(1, 9))
    assert list(r["hotel"]) == list(novas["hotel"])


def test_reserva_invalida_e_recusada_com_motivo(modelo, cfg):
    reserva = _reservas_novas(1).iloc[0].to_dict() | {"adults": 0, "children": 0, "babies": 0}
    with pytest.raises(ValueError, match="sem hóspedes"):
        prever.prever_reserva(reserva, modelo, cfg=cfg)


def test_exemplos_do_repositorio_sao_reservas_validas_sem_rotulo(modelo, cfg):
    for nome in ("reserva_exemplo.json", "reservas_exemplo.json"):
        reservas = prever.ler_reservas(RAIZ / "exemplos" / nome)
        for r in reservas:
            assert ALVO not in r and not set(COLUNAS_POS_RESERVA) & set(r)
            assert set(COLUNAS_RESERVA) <= set(r)
        assert len(prever.prever_reserva(reservas, modelo, cfg=cfg)) == len(reservas)


def test_json_invalido_e_recusado(tmp_path):
    arquivo = tmp_path / "x.json"
    arquivo.write_text("[1, 2]")
    with pytest.raises(ValueError, match="objeto JSON"):
        prever.ler_reservas(arquivo)


# ------------------------------------------------------------------------------------- linha de comando

def _projeto_com_vigente(raiz: Path, modelo, cfg) -> None:
    arquivo = guardar_modelo(modelo, "c0", "A", raiz)
    atualizar_manifesto({
        "cycle_id": "c0", "data_version": "dv1", "decisao": "promovido", "motivo": "teste",
        "regra": {"metrica": "f1_macro"}, "valor_vencedor": 0.7, "valor_vigente": None,
        "candidato": {"modelo": "A", "algoritmo": "logistic_regression", "params": {}, "features": ["lead_time"],
                      "arquivo": arquivo.name},
    }, cfg, raiz)


def test_cli_usa_o_vigente_ou_outro_manifesto(modelo, cfg, tmp_path, capsys):
    _projeto_com_vigente(tmp_path, modelo, cfg)
    entrada = _json_reserva(tmp_path)
    assert prever.main(["--json", str(entrada)], raiz=tmp_path) == 0
    assert "Reserva 1" in capsys.readouterr().out

    copia = tmp_path / "manifesto_copia.json"     # como models/modelo_vigente_actions.json
    (tmp_path / "models" / "modelo_vigente.json").rename(copia)
    assert prever.main(["--json", str(entrada), "--manifesto", str(copia)], raiz=tmp_path) == 0
    assert "Modelo vigente: A do ciclo c0" in capsys.readouterr().out


def test_cli_sem_vigente_explica_o_que_fazer(tmp_path, capsys):
    assert prever.main(["--json", str(_json_reserva(tmp_path))], raiz=tmp_path) == 1
    assert "Ainda não há modelo vigente" in capsys.readouterr().err


def test_demonstracao_e_gravada_fora_da_conferencia(modelo, cfg, tmp_path):
    r = prever.prever_reserva(_reservas_novas(2), modelo, {"cycle_id": "c0", "modelo": "A"}, cfg)
    caminho = prever.salvar_demonstracao(r, {"cycle_id": "c5", "data_version": "dv5"}, tmp_path)
    assert caminho.name == "demonstracao_c5.csv" and not caminho.name.startswith("previsoes_")
    assert len(pd.read_csv(caminho)) == 2
