"""F3: os três métodos, o contrato com a F4 e a ausência de vazamento entre treino e validação."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.avaliacao import criar_cv
from src.features import calcular_rankings, calcular_rankings_por_fold, salvar_rankings, tabela_rankings
from src.preprocessamento import limpar_e_engenhar, lista_features

METODOS = {"selectkbest", "random_forest", "rfe"}


@pytest.fixture(scope="module")
def cfg_rapido(cfg) -> dict:
    """Config real com floresta menor, para os testes rodarem em segundos."""
    c = copy.deepcopy(cfg)
    c["selecao"]["metodos"]["random_forest"].update(n_estimators=20, max_depth=4)
    c["selecao"]["k_top"] = 3
    return c


@pytest.fixture()
def X_y() -> tuple[pd.DataFrame, pd.Series]:
    """Só ``sinal`` e ``binaria_sinal`` explicam o alvo; as demais colunas são ruído."""
    rng = np.random.default_rng(0)
    n = 1500
    sinal = rng.normal(size=n)
    binaria = rng.integers(0, 2, n).astype(float)
    y = pd.Series((sinal + 2 * binaria + rng.normal(scale=0.3, size=n) > 1).astype(int), name="alvo")
    X = pd.DataFrame({"sinal": sinal, "binaria_sinal": binaria, **{f"ruido_{i}": rng.normal(size=n) for i in range(6)}})
    return X, y


@pytest.fixture()
def dev_sintetico(df_bruto_multilote, cfg) -> pd.DataFrame:
    return limpar_e_engenhar(df_bruto_multilote, cfg)[0]


def test_tres_metodos_com_o_mesmo_indice_e_scores_validos(X_y, cfg_rapido):
    X, y = X_y
    rankings = calcular_rankings(X, y, cfg_rapido)
    assert set(rankings) == METODOS
    for score in rankings.values():
        assert list(score.index) == list(X.columns)
        assert np.isfinite(score).all() and (score >= 0).all()


def test_features_com_sinal_ficam_acima_do_ruido_em_todos_os_metodos(X_y, cfg_rapido):
    X, y = X_y
    c = copy.deepcopy(cfg_rapido)
    c["selecao"]["k_top"] = 2  # exatamente as duas com sinal; com mais, o RFE teria de manter ruído
    for metodo, score in calcular_rankings(X, y, c).items():
        assert score[["sinal", "binaria_sinal"]].min() > score.filter(like="ruido").max(), metodo


def test_rfe_mantem_exatamente_k_top(X_y, cfg_rapido):
    X, y = X_y
    rfe = calcular_rankings(X, y, cfg_rapido)["rfe"]
    assert (rfe == 1.0).sum() == cfg_rapido["selecao"]["k_top"]


def test_k_top_maior_que_o_numero_de_features_nao_quebra(X_y, cfg_rapido):
    X, y = X_y
    c = copy.deepcopy(cfg_rapido)
    c["selecao"]["k_top"] = 500
    assert (calcular_rankings(X, y, c)["rfe"] == 1.0).all()


def test_mesma_seed_mesmos_rankings(X_y, cfg_rapido):
    X, y = X_y
    a, b = calcular_rankings(X, y, cfg_rapido), calcular_rankings(X, y, cfg_rapido)
    for metodo in METODOS:
        assert_series_equal(a[metodo], b[metodo])


def test_metodo_nao_suportado_falha_em_vez_de_trocar_em_silencio(X_y, cfg_rapido):
    X, y = X_y
    c = copy.deepcopy(cfg_rapido)
    c["selecao"]["metodos"]["selectkbest"]["score_func"] = "f_classif"
    with pytest.raises(ValueError, match="score_func"):
        calcular_rankings(X, y, c)


def test_indices_desalinhados_falham(X_y, cfg_rapido):
    X, y = X_y
    with pytest.raises(ValueError, match="mesmo índice"):
        calcular_rankings(X, y.iloc[::-1], cfg_rapido)


def test_por_fold_entrega_o_contrato_da_f4(dev_sintetico, cfg_rapido):
    X_bruto = dev_sintetico[lista_features(cfg_rapido)]
    y = dev_sintetico[cfg_rapido["avaliacao"]["alvo"]]
    rankings, por_fold = calcular_rankings_por_fold(X_bruto, y, cfg_rapido)
    assert set(rankings) == METODOS
    indices = [list(s.index) for s in rankings.values()]
    assert indices[0] == indices[1] == indices[2]
    assert not any(s.isna().any() for s in rankings.values())
    assert sorted(por_fold["fold"].unique()) == list(range(1, cfg_rapido["avaliacao"]["cv"]["n_splits"] + 1))


def test_validacao_do_ultimo_fold_nao_influencia_a_selecao(dev_sintetico, cfg_rapido):
    """Trocar X e y em linhas fora de todo treino não pode mudar nada: prova que só o treino é usado."""
    X_bruto = dev_sintetico[lista_features(cfg_rapido)]
    y = dev_sintetico[cfg_rapido["avaliacao"]["alvo"]]
    _, validacao = list(criar_cv(cfg_rapido).split(X_bruto))[-1]

    X_alterado, y_alterado = X_bruto.copy(), y.copy()
    linhas = X_bruto.index[validacao]
    X_alterado.loc[linhas, "lead_time"] = 100_000
    X_alterado.loc[linhas, "country"] = "CATEGORIA_NOVA"
    y_alterado.loc[linhas] = 1 - y_alterado.loc[linhas]

    _, original = calcular_rankings_por_fold(X_bruto, y, cfg_rapido)
    _, alterado = calcular_rankings_por_fold(X_alterado, y_alterado, cfg_rapido)
    assert_frame_equal(original, alterado)


def test_treino_alterado_muda_a_selecao(dev_sintetico, cfg_rapido):
    """Controle do teste acima: a mesma troca feita no treino precisa ser detectada."""
    X_bruto = dev_sintetico[lista_features(cfg_rapido)]
    y = dev_sintetico[cfg_rapido["avaliacao"]["alvo"]]
    treino, _ = list(criar_cv(cfg_rapido).split(X_bruto))[0]
    y_alterado = y.copy()
    y_alterado.iloc[treino] = 1 - y_alterado.iloc[treino].to_numpy()
    X_alterado = X_bruto.copy()
    X_alterado.iloc[treino, X_alterado.columns.get_loc("lead_time")] = 100_000

    _, original = calcular_rankings_por_fold(X_bruto, y, cfg_rapido)
    _, alterado = calcular_rankings_por_fold(X_alterado, y_alterado, cfg_rapido)
    assert not original["score"].equals(alterado["score"])


def test_feature_ausente_em_um_fold_entra_na_uniao_e_conta_zero(dev_sintetico, cfg_rapido):
    """Categoria que só existe no treino do último fold: 1 fold presente, média = score / n_folds."""
    X_bruto = dev_sintetico[lista_features(cfg_rapido)].copy()
    y = dev_sintetico[cfg_rapido["avaliacao"]["alvo"]]
    divisoes = list(criar_cv(cfg_rapido).split(X_bruto))
    so_no_ultimo = np.setdiff1d(divisoes[-1][0], divisoes[-2][0])
    X_bruto.iloc[so_no_ultimo, X_bruto.columns.get_loc("country")] = "TARDIO"

    rankings, por_fold = calcular_rankings_por_fold(X_bruto, y, cfg_rapido)
    tabela = tabela_rankings(rankings, por_fold)

    assert tabela.loc["country_TARDIO", "folds_presente"] == 1
    no_fold = por_fold.query("feature == 'country_TARDIO' and metodo == 'random_forest'")["score"].item()
    assert rankings["random_forest"]["country_TARDIO"] == pytest.approx(no_fold / len(divisoes))


def test_salvar_rankings_identifica_o_ciclo(dev_sintetico, cfg_rapido, tmp_path: Path):
    X_bruto = dev_sintetico[lista_features(cfg_rapido)]
    y = dev_sintetico[cfg_rapido["avaliacao"]["alvo"]]
    rankings, por_fold = calcular_rankings_por_fold(X_bruto, y, cfg_rapido)
    ctx = {"cycle_id": "ciclo-x", "data_version": "dv001-abc"}
    caminhos = salvar_rankings(tabela_rankings(rankings, por_fold), por_fold, ctx, tmp_path)

    salvo = pd.read_csv(caminhos["rankings"])
    assert caminhos["rankings"].name == "rankings_ciclo-x.csv"
    assert set(salvo["cycle_id"]) == {"ciclo-x"} and set(salvo["data_version"]) == {"dv001-abc"}
    assert {"score_rfe", "posicao_rfe", "folds_presente"} <= set(salvo.columns)
    assert len(pd.read_csv(caminhos["por_fold"])) == len(por_fold)


def test_notebook_da_selecao_nao_abre_o_conjunto_de_teste():
    caminho = Path(__file__).resolve().parents[1] / "notebooks" / "02_votacao_features.ipynb"
    celulas = json.loads(caminho.read_text(encoding="utf-8"))["cells"]
    codigo = "\n".join("".join(c["source"]) for c in celulas if c["cell_type"] == "code")
    assert 'carregar_particao("dev"' in codigo
    assert "teste" not in codigo.lower()
