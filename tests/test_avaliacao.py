import numpy as np
import pytest

from src.avaliacao import criar_cv, melhorou, metricas_classificacao


def test_metricas_perfeitas_e_chaves():
    y = np.array([0, 1, 1, 0, 1, 0])
    m = metricas_classificacao(y, y, y_score=y.astype(float))
    assert m["f1_macro"] == 1.0 and m["roc_auc"] == 1.0 and m["recall_cancelamento"] == 1.0
    assert set(m) == {"f1_macro", "balanced_accuracy", "recall_cancelamento", "precision_cancelamento", "roc_auc", "pr_auc"}


def test_metricas_sem_score_omitem_auc():
    m = metricas_classificacao([0, 1, 1, 0], [0, 1, 0, 0])
    assert "roc_auc" not in m and 0 < m["f1_macro"] < 1


def test_cv_e_cronologico(cfg):
    cv = criar_cv(cfg)
    X = np.arange(1000).reshape(-1, 1)
    folds = list(cv.split(X))
    assert len(folds) == cfg["avaliacao"]["cv"]["n_splits"]
    for treino, val in folds:
        assert treino.max() < val.min()  # validação sempre no futuro do treino


def test_cv_tipo_desconhecido(cfg):
    ruim = {"avaliacao": {"cv": {"tipo": "kfold", "n_splits": 3}}}
    with pytest.raises(ValueError):
        criar_cv(ruim)


@pytest.mark.parametrize(
    "novo,base,direcao,minimo,esperado",
    [
        (0.80, 0.78, "maximizar", 0.005, True),
        (0.782, 0.78, "maximizar", 0.005, False),  # melhora menor que o limiar
        (0.78, 0.78, "maximizar", 0.0, False),     # empate nunca promove
        (0.70, 0.78, "maximizar", 0.0, False),
        (0.40, 0.45, "minimizar", 0.02, True),     # métrica de erro
        (0.44, 0.45, "minimizar", 0.02, False),
    ],
)
def test_melhorou(novo, base, direcao, minimo, esperado):
    assert melhorou(novo, base, direcao, minimo) is esperado


def test_melhorou_direcao_invalida():
    with pytest.raises(ValueError):
        melhorou(1, 0, "subir")
