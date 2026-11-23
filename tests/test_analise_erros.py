"""Matriz de confusão e análise de erros do notebook 04 (descreve o modelo; não muda a decisão)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import analise_erros
from src.modelagem import construir_pipeline
from src.preprocessamento import limpar_e_engenhar, lista_features
from tests.helpers import gerar_reservas


def test_matriz_e_resumo_contam_cada_tipo_de_erro():
    y = [0, 0, 0, 1, 1, 1, 1]
    p = [0, 1, 1, 0, 1, 1, 1]
    m = analise_erros.matriz_confusao(y, p)
    assert m.loc["não cancelou", "não cancelou"] == 1 and m.loc["não cancelou", "cancelou"] == 2
    assert m.loc["cancelou", "não cancelou"] == 1 and m.loc["cancelou", "cancelou"] == 3
    r = analise_erros.resumo_confusao(y, p)
    assert (r["falsos_positivos"], r["falsos_negativos"], r["n"]) == (2, 1, 7)
    assert r["taxa_erro"] == pytest.approx(3 / 7)
    assert r["taxa_falso_positivo"] == pytest.approx(2 / 3) and r["taxa_falso_negativo"] == pytest.approx(1 / 4)


def test_matriz_tem_as_duas_classes_mesmo_sem_previsao_positiva():
    m = analise_erros.matriz_confusao([0, 1, 1], [0, 0, 0])
    assert m.shape == (2, 2) and m["cancelou"].sum() == 0


def test_faixas_de_lead_time():
    faixas = analise_erros.faixa_lead_time(pd.Series([0, 6, 7, 29, 30, 400]), [0, 7, 30])
    assert list(faixas) == ["0-6", "0-6", "7-29", "7-29", "30+", "30+"]


def test_erros_por_grupo_soma_os_erros_e_junta_grupos_pequenos():
    grupos = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"], name="hotel")
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    p = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0])
    total = int(((y == 0) & (p == 1)).sum() + ((y == 1) & (p == 0)).sum())
    t = analise_erros.erros_por_grupo(grupos, y, p, total, min_reservas=3).set_index("valor")
    assert set(t.index) == {"a", "b", "outros"}
    assert int((t["falsos_positivos"] + t["falsos_negativos"]).sum()) == total
    assert t["parcela_dos_erros"].sum() == pytest.approx(1.0)
    assert t.loc["a", "taxa_erro"] == pytest.approx(2 / 6)


def test_analise_completa_salva_evidencias(cfg, tmp_path):
    brutos = pd.concat([gerar_reservas(150, 2016, m, seed=m).assign(lote_id=f"2016-{m:02d}") for m in range(1, 6)], ignore_index=True)
    brutos["linha_origem"] = range(len(brutos))
    df, _ = limpar_e_engenhar(brutos, cfg)
    dev, teste = df.iloc[:600], df.iloc[600:]
    modelo = construir_pipeline("A", ["lead_time", "adr"], cfg).fit(dev[lista_features(cfg)], dev[cfg["avaliacao"]["alvo"]])

    analise = analise_erros.analisar_erros(modelo, teste, cfg)
    assert analise["resumo"]["n"] == len(teste) and int(analise["matriz"].to_numpy().sum()) == len(teste)
    assert set(analise["por_grupo"]["grupo"]) == set(cfg["analise_erros"]["grupos"]) | {"faixa_lead_time"}
    assert len(analise["maiores_taxas_erro"]) <= cfg["analise_erros"]["n_destaques"]

    caminhos = analise_erros.salvar_analise_erros(analise, {"cycle_id": "c1", "data_version": "dv1"}, cfg, "A do ciclo c1", tmp_path)
    assert all(c.exists() for c in caminhos.values())
    registro = json.loads(caminhos["json"].read_text(encoding="utf-8"))
    assert registro["cycle_id"] == "c1" and registro["modelo_analisado"] == "A do ciclo c1"
    assert registro["falsos_positivos"] + registro["falsos_negativos"] + registro["verdadeiros_positivos"] + registro["verdadeiros_negativos"] == len(teste)
    metricas = analise_erros.metricas_para_mlflow(analise)
    assert metricas["teste_em_uso_n"] == len(teste) and all(isinstance(v, float) for v in metricas.values())
