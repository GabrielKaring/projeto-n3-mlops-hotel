import copy

import numpy as np
import pandas as pd
import pytest

from src.preprocessamento import (
    Winsorizador,
    construir_preprocessador,
    dividir_temporal,
    limpar_e_engenhar,
    lista_features,
    remover_duplicatas,
    validar_dados_brutos,
)
from src.schema import ALVO, construir_data_chegada
from tests.helpers import gerar_reservas


# ----------------------------------------------------------------------------- limpeza


def test_limpeza_remove_linhas_invalidas_e_conta(cfg, df_bruto_multilote):
    df = df_bruto_multilote.copy()
    df.loc[0, ["adults", "children", "babies"]] = [0, 0.0, 0]  # sem hóspedes
    df.loc[1, "adr"] = -5.0  # adr negativo
    limpo, rel = limpar_e_engenhar(df, cfg)
    assert rel["regras_dominio"]["sem_hospedes"] >= 1 and rel["regras_dominio"]["adr_negativo"] == 1
    assert rel["n_entrada"] - rel["n_saida"] == rel["removidas_total"]
    assert (limpo["adr"] >= 0).all() and (limpo["total_hospedes"] > 0).all()


def test_nao_sobra_ausente_nas_features(cfg, df_bruto_multilote):
    limpo, rel = limpar_e_engenhar(df_bruto_multilote, cfg)
    assert limpo[lista_features(cfg)].isna().sum().sum() == 0
    assert rel["imputados"]["agent"] > 0  # o gerador sintético cria agentes ausentes


def test_colunas_de_vazamento_nao_chegam_ao_modelo(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    proibidas = {"reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes"}
    assert proibidas.isdisjoint(limpo.columns)
    assert ALVO in limpo.columns and ALVO not in lista_features(cfg)


def test_features_derivadas_corretas(cfg):
    df = gerar_reservas(5, 2016, 3)
    df.insert(0, "linha_origem", range(5))
    df.insert(0, "lote_id", "lote_2016-03")
    df.loc[0, ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "adr"]] = [2, 3, 2, 1.0, 1, 100.0]
    df.loc[0, ["previous_cancellations", "previous_bookings_not_canceled"]] = [1, 3]
    limpo, _ = limpar_e_engenhar(df, cfg)
    linha = limpo[(limpo["linha_origem"] == 0)].iloc[0]
    assert linha["total_noites"] == 5 and linha["total_hospedes"] == 4
    assert linha["tem_criancas"] == 1 and linha["valor_estimado_estadia"] == 500.0
    assert linha["razao_cancel_previo"] == pytest.approx(0.25)


def test_saida_ordenada_cronologicamente(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote.sample(frac=1, random_state=1), cfg)
    assert limpo["arrival_date"].is_monotonic_increasing


def test_validacao_barra_dado_invalido(cfg):
    ruim = gerar_reservas(10, 2016, 1)
    ruim.loc[0, "arrival_date_month"] = "Janvier"
    with pytest.raises(ValueError, match="Meses"):
        validar_dados_brutos(ruim)
    with pytest.raises(ValueError, match="Meses"):
        construir_data_chegada(ruim)
    ruim2 = gerar_reservas(10, 2016, 1)
    ruim2.loc[0, ALVO] = 7
    with pytest.raises(ValueError, match="apenas 0 e 1"):
        validar_dados_brutos(ruim2)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        validar_dados_brutos(ruim2.drop(columns=["lead_time"]))


# ----------------------------------------------------------------------------- duplicatas


def _lote(nome, base, copias):
    df = pd.concat([base] * copias, ignore_index=True)
    df.insert(0, "linha_origem", range(len(df)))
    df.insert(0, "lote_id", nome)
    return df


def test_duplicatas_entre_lotes_mantem_copias_do_mesmo_lote_e_remove_reentrega():
    linha = gerar_reservas(1, 2016, 1)
    a = _lote("A", linha, copias=5)  # 5 reservas idênticas legítimas (vários quartos)
    b = _lote("B", linha, copias=3)  # reentrega de 3 delas
    todos = pd.concat([a, b], ignore_index=True)
    assert len(remover_duplicatas(todos, "entre_lotes")) == 5
    assert len(remover_duplicatas(a, "entre_lotes")) == 5
    assert len(remover_duplicatas(todos, "todas")) == 1
    assert len(remover_duplicatas(todos, "nenhuma")) == 8
    with pytest.raises(ValueError):
        remover_duplicatas(todos, "outra")


def test_etl_com_reentrega_de_lote_nao_altera_resultado(cfg, df_bruto_multilote):
    base, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    reentrega = df_bruto_multilote[df_bruto_multilote["lote_id"] == "lote_2016-03"].copy()
    reentrega["lote_id"] = "lote_reenviado"
    com_reentrega, rel = limpar_e_engenhar(pd.concat([df_bruto_multilote, reentrega], ignore_index=True), cfg)
    assert len(com_reentrega) == len(base)
    assert rel["duplicatas_removidas"] == len(reentrega)


# ----------------------------------------------------------------------------- partição temporal


def test_particao_cronologica_sem_sobreposicao(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    dev, teste = dividir_temporal(limpo, 0.20)
    assert dev["arrival_date"].max() < teste["arrival_date"].min()  # nenhuma data nos dois lados
    assert len(dev) + len(teste) == len(limpo)
    ids_dev = set(zip(dev["lote_id"], dev["linha_origem"]))
    ids_teste = set(zip(teste["lote_id"], teste["linha_origem"]))
    assert ids_dev.isdisjoint(ids_teste)
    assert 0.15 <= len(teste) / len(limpo) <= 0.25
    assert dev["arrival_date"].is_monotonic_increasing


def test_particao_rejeita_fracao_invalida(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    for f in (0, 1, 1.5):
        with pytest.raises(ValueError):
            dividir_temporal(limpo, f)


# ----------------------------------------------------------------------------- ausência de vazamento no pré-processador


def test_winsorizador_aprende_limites_so_no_fit():
    treino = pd.DataFrame({"x": np.arange(0, 101, dtype=float)})
    w = Winsorizador(q_inf=0.0, q_sup=0.9).fit(treino)
    assert w.limite_sup_[0] == pytest.approx(90.0)
    fora = pd.DataFrame({"x": [-50.0, 50.0, 1e6, np.nan]})
    saida = w.transform(fora)[:, 0]
    assert saida[0] == 0.0 and saida[1] == 50.0 and saida[2] == 90.0 and np.isnan(saida[3])


def test_preprocessador_usa_somente_estatisticas_do_treino(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    dev, teste = dividir_temporal(limpo, 0.20)
    feats = lista_features(cfg)
    # Teste deliberadamente extremo: se o ajuste olhasse para ele, a média mudaria muito.
    teste = teste.copy()
    teste["lead_time"] = 100000

    pre = construir_preprocessador(cfg)
    X_dev = pre.fit_transform(dev[feats])
    X_teste = pre.transform(teste[feats])

    escala = pre.named_transformers_["num_cauda"].named_steps["escala"]
    nomes = list(pre.named_transformers_["num_cauda"].named_steps["winsor"].get_feature_names_out())
    i = nomes.index("lead_time")
    w = pre.named_transformers_["num_cauda"].named_steps["winsor"]
    esperado = np.clip(dev["lead_time"].to_numpy(float), w.limite_inf_[i], w.limite_sup_[i]).mean()
    assert escala.mean_[i] == pytest.approx(esperado)  # média calculada só no dev
    assert X_dev["lead_time"].mean() == pytest.approx(0.0, abs=1e-9)
    assert X_teste["lead_time"].max() == pytest.approx((w.limite_sup_[i] - escala.mean_[i]) / escala.scale_[i])  # teste foi cortado no limite do treino
    assert list(X_dev.columns) == list(X_teste.columns)
    assert X_dev.isna().sum().sum() == 0 and X_teste.isna().sum().sum() == 0


def test_preprocessador_tolera_categoria_nova_e_ausente_em_dado_futuro(cfg, df_bruto_multilote):
    limpo, _ = limpar_e_engenhar(df_bruto_multilote, cfg)
    dev, teste = dividir_temporal(limpo, 0.20)
    feats = lista_features(cfg)
    pre = construir_preprocessador(cfg)
    pre.fit(dev[feats])
    futuro = teste[feats].copy()
    futuro["country"] = "ZZZ"  # país nunca visto
    futuro.loc[futuro.index[0], "adr"] = np.nan  # ausente novo
    X = pre.transform(futuro)
    assert X.isna().sum().sum() == 0 and len(X) == len(futuro)


def test_config_nao_coloca_alvo_nem_meta_nas_features(cfg):
    feats = set(lista_features(cfg))
    assert ALVO not in feats and feats.isdisjoint(cfg["features"]["meta"])
    assert len(feats) == len(lista_features(cfg))  # sem repetidas


def test_config_nao_aceita_estrategia_inexistente(cfg, df_bruto_multilote):
    ruim = copy.deepcopy(cfg)
    ruim["etl"]["duplicatas"]["estrategia"] = "talvez"
    with pytest.raises(ValueError, match="desconhecida"):
        limpar_e_engenhar(df_bruto_multilote, ruim)


def test_country_nao_chega_ao_modelo(df_bruto_multilote, cfg):
    """A nacionalidade correta costuma ser conhecida só no check-in (docs/etapa1, seção 3)."""
    df, rel = limpar_e_engenhar(df_bruto_multilote, cfg)
    assert "country" not in lista_features(cfg) and "country" not in df.columns
    assert "country" in rel["colunas_removidas"]
