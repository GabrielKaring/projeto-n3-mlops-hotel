"""Protocolo de avaliação único para todo o pipeline (Etapa 1).

Todos os candidatos (Modelo A, Modelo B e o modelo vigente) devem ser medidos com estas funções e
com o mesmo ``criar_cv``, para que a comparação seja justa e reprodutível.
"""
from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


def metricas_classificacao(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Métricas definidas na config. ``y_score`` (probabilidade da classe 1) habilita ROC-AUC e PR-AUC."""
    m = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_cancelamento": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_cancelamento": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: float(v) for k, v in m.items()}


def criar_cv(cfg: dict) -> TimeSeriesSplit:
    """Validação cruzada cronológica com janela expansiva. O dev deve estar ordenado por data."""
    cv = cfg["avaliacao"]["cv"]
    if cv["tipo"] != "time_series_split":
        raise ValueError(f"Tipo de CV não suportado: {cv['tipo']}")
    return TimeSeriesSplit(n_splits=cv["n_splits"])


def melhorou(novo: float, base: float, direcao: str = "maximizar", min_melhoria: float = 0.0) -> bool:
    """True se ``novo`` supera ``base`` em pelo menos ``min_melhoria`` na direção da métrica.

    Empate nunca conta como melhora: com ``min_melhoria=0`` o ganho precisa ser estritamente positivo.
    """
    if direcao not in ("maximizar", "minimizar"):
        raise ValueError("direcao deve ser 'maximizar' ou 'minimizar'")
    ganho = (novo - base) if direcao == "maximizar" else (base - novo)
    return bool(ganho > 1e-12 and ganho >= min_melhoria - 1e-12)
