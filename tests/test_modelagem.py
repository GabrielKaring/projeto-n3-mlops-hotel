"""F5: os dois modelos treinam, prevêem e são avaliados no mesmo protocolo."""
from __future__ import annotations

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.avaliacao import dividir_cv
from src.modelagem import (
    SeletorFeatures,
    avaliar_cv,
    construir_pipeline,
    eleger_por_fold,
    montar_resultados,
    salvar_modelos,
    salvar_resultados,
    treinar_e_avaliar,
)
from src.preprocessamento import construir_preprocessador, limpar_e_engenhar, lista_features
from tests.helpers import gerar_reservas


@pytest.fixture(scope="module")
def cfg_rapido(cfg) -> dict:
    c = copy.deepcopy(cfg)
    c["modelos"]["B"]["params"].update(n_estimators=20, max_depth=4)
    c["selecao"]["k_top"] = 3
    c["selecao"]["votacao"]["min_features"] = 2
    return c


@pytest.fixture(scope="module")
def dev(cfg_rapido) -> pd.DataFrame:
    brutos = pd.concat([gerar_reservas(150, 2016, m, seed=m).assign(lote_id=f"2016-{m:02d}") for m in range(1, 7)], ignore_index=True)
    brutos["linha_origem"] = range(len(brutos))
    df, _ = limpar_e_engenhar(brutos, cfg_rapido)
    return df.sort_values(cfg_rapido["avaliacao"]["coluna_tempo"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def eleitas(dev, cfg_rapido) -> list[str]:
    nomes = construir_preprocessador(cfg_rapido).fit(dev[lista_features(cfg_rapido)]).get_feature_names_out()
    return ["lead_time", "adr", "deposit_type_Non Refund", "hotel_City Hotel"] + [n for n in nomes if n.startswith("meal_")][:1]


def test_seletor_mantem_ordem_e_zera_coluna_ausente():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    seletor = SeletorFeatures(["b", "nao_existe", "a"]).fit(X)
    saida = seletor.transform(X)
    assert list(saida.columns) == ["b", "nao_existe", "a"]
    assert (saida["nao_existe"] == 0).all()
    assert seletor.ausentes_ == ["nao_existe"]


def test_seletor_recusa_lista_vazia():
    with pytest.raises(ValueError):
        SeletorFeatures([]).fit(pd.DataFrame({"a": [1.0]}))


@pytest.mark.parametrize("nome", ["A", "B"])
def test_modelos_treinam_e_preveem_com_as_mesmas_colunas(nome, dev, eleitas, cfg_rapido):
    pipe = construir_pipeline(nome, eleitas, cfg_rapido)
    pipe.fit(dev[lista_features(cfg_rapido)], dev[cfg_rapido["avaliacao"]["alvo"]])
    proba = pipe.predict_proba(dev[lista_features(cfg_rapido)].head(20))[:, 1]
    assert proba.shape == (20,) and ((proba >= 0) & (proba <= 1)).all()
    assert list(pipe.named_steps["modelo"].feature_names_in_) == eleitas


def test_a_e_b_sao_algoritmos_diferentes(eleitas, cfg_rapido):
    a = construir_pipeline("A", eleitas, cfg_rapido).named_steps["modelo"]
    b = construir_pipeline("B", eleitas, cfg_rapido).named_steps["modelo"]
    assert type(a) is not type(b)
    assert a.random_state == b.random_state == cfg_rapido["projeto"]["seed"]


def test_modelo_desconhecido_falha(eleitas, cfg_rapido):
    with pytest.raises(ValueError):
        construir_pipeline("C", eleitas, cfg_rapido)


def test_a_e_b_usam_os_mesmos_folds(dev, eleitas, cfg_rapido):
    n = cfg_rapido["avaliacao"]["cv"]["n_splits"]
    res = {nome: avaliar_cv(construir_pipeline(nome, eleitas, cfg_rapido), dev, cfg_rapido) for nome in ("A", "B")}
    for r in res.values():
        assert [f["fold"] for f in r["por_fold"]] == list(range(1, n + 1))
        assert set(r["media"]) == set(r["desvio"]) >= {"f1_macro", "roc_auc"}


def test_cv_nao_ve_a_validacao_no_treino(dev, eleitas, cfg_rapido, monkeypatch):
    """Cada fold ajusta só com linhas anteriores à validação."""
    vistos = []
    original = SeletorFeatures.fit

    def espiao(self, X, y=None):
        vistos.append(X.index)
        return original(self, X, y)

    monkeypatch.setattr(SeletorFeatures, "fit", espiao)
    avaliar_cv(construir_pipeline("A", eleitas, cfg_rapido), dev, cfg_rapido)
    tempo = dev[cfg_rapido["avaliacao"]["coluna_tempo"]]
    for idx_treino, (treino, valid) in zip(vistos, dividir_cv(tempo, cfg_rapido)):
        assert set(idx_treino) == set(dev.index[treino])
        assert tempo.iloc[treino].max() <= tempo.iloc[valid].min()


def test_feature_ausente_no_fold_e_anotada(dev, eleitas, cfg_rapido):
    res = avaliar_cv(construir_pipeline("A", eleitas + ["agent_inexistente"], cfg_rapido), dev, cfg_rapido)
    assert all("agent_inexistente" in v for v in res["features_ausentes"].values())
    assert len(res["features_ausentes"]) == cfg_rapido["avaliacao"]["cv"]["n_splits"]


def test_treinar_e_avaliar_deixa_pipeline_ajustado(dev, eleitas, cfg_rapido):
    pipe = construir_pipeline("A", eleitas, cfg_rapido)
    res = treinar_e_avaliar(pipe, dev, cfg_rapido)
    assert "f1_macro" in res["media"]
    pipe.predict(dev[lista_features(cfg_rapido)].head(5))


def test_features_por_fold_troca_o_conjunto(dev, eleitas, cfg_rapido):
    n = cfg_rapido["avaliacao"]["cv"]["n_splits"]
    por_fold = {k: eleitas[:2] + ["coluna_extra"] for k in range(1, n + 1)}
    res = avaliar_cv(construir_pipeline("A", eleitas, cfg_rapido), dev, cfg_rapido, features_por_fold=por_fold)
    assert all(v == ["coluna_extra"] for v in res["features_ausentes"].values())


def test_eleger_por_fold_vota_cada_fold_separado(cfg_rapido):
    linhas = []
    for fold, melhor in [(1, "x"), (2, "y")]:
        for metodo in cfg_rapido["selecao"]["metodos"]:
            for feat, score in {"x": 1.0, "y": 1.0, "z": 0.1, "w": 0.05}.items():
                linhas.append({"fold": fold, "metodo": metodo, "feature": feat, "score": score * (3 if feat == melhor else 1)})
    eleitas = eleger_por_fold(pd.DataFrame(linhas), cfg_rapido)
    assert eleitas[1][0] == "x" and eleitas[2][0] == "y"


def test_salvar_modelos_e_resultados(tmp_path, dev, eleitas, cfg_rapido):
    ctx = {"cycle_id": "teste-ciclo", "data_version": "dv000-abc"}
    cfg_tmp = copy.deepcopy(cfg_rapido)
    cfg_tmp["dados"]["dir_processed"] = str(tmp_path / "processed")
    pipes = {nome: construir_pipeline(nome, eleitas, cfg_tmp) for nome in ("A", "B")}
    avaliacoes = {nome: treinar_e_avaliar(p, dev, cfg_tmp) for nome, p in pipes.items()}
    caminhos = salvar_modelos(pipes, ctx, raiz=tmp_path)
    resultados = montar_resultados(ctx, cfg_tmp, eleitas, avaliacoes, sensibilidade=avaliacoes, caminhos_modelos=caminhos)
    salvos = salvar_resultados(resultados, cfg_tmp, raiz=tmp_path)

    lido = json.loads(salvos["resultados"].read_text(encoding="utf-8"))
    assert lido["cycle_id"] == "teste-ciclo" and lido["features_eleitas"] == eleitas
    assert set(lido["modelos"]) == {"A", "B"}
    assert lido["modelos"]["A"]["sensibilidade_eleicao_por_fold"]["diferenca_metrica_principal"] == pytest.approx(0.0)
    assert salvos["evidencia"].name == "resultados_teste-ciclo.json"

    recarregado = joblib.load(caminhos["B"])
    X = dev[lista_features(cfg_tmp)].head(10)
    np.testing.assert_allclose(recarregado.predict_proba(X), pipes["B"].predict_proba(X))
