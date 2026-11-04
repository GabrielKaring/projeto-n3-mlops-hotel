"""Etapa 5 do enunciado: eleição do vencedor e promoção controlada.

STATUS: a implementar (ver docs/divisao_de_tarefas.md, critério 8).

Contrato de entrada (notebook 04):
  * Métricas dos modelos A e B do ciclo (notebook 03) e o manifesto ``models/modelo_vigente.json``
    recuperado do armazenamento persistente (o runner do GitHub Actions é efêmero).
  * Regra em ``cfg["promocao"]``: métrica, direção, melhoria mínima e piso. Já fixada antes de rodar.

Regras do enunciado:
  * O vencedor do ciclo é comparado ao vigente com protocolo compatível (mesmos dados de validação).
  * Só promove se cumprir a regra; caso contrário o vigente é preservado.
  * Modelos anteriores nunca são apagados.
  * No primeiro ciclo (sem vigente) o vencedor é promovido se passar no piso.
  * Use ``src.avaliacao.melhorou`` para a comparação.

Contrato de saída:
  * ``models/modelo_vigente.json``: identificação do modelo, data_version, features, métrica, run_id,
    cycle_id e decisão (``promovido`` ou ``mantido``) com o motivo.
"""
from __future__ import annotations


def eleger_vencedor(resultados: dict[str, dict], cfg: dict) -> str:
    """Retorna ``'A'`` ou ``'B'`` conforme a métrica principal da config."""
    raise NotImplementedError("TODO(critério 8): comparar automaticamente os dois candidatos")


def decidir_promocao(vencedor: dict, vigente: dict | None, cfg: dict) -> dict:
    """Retorna ``{"decisao": "promovido"|"mantido", "motivo": str, ...}`` aplicando a regra da config."""
    raise NotImplementedError("TODO(critério 8): regra de promoção com limiar de melhoria")


def atualizar_manifesto(decisao: dict, cfg: dict) -> None:
    """Grava ``models/modelo_vigente.json`` preservando o histórico de modelos anteriores."""
    raise NotImplementedError("TODO(critério 8): manifesto auditável do modelo vigente")
