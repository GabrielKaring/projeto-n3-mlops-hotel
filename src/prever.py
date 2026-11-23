"""Demonstração de uso: probabilidade de cancelamento de uma reserva nova, com o modelo vigente.

Entrada: uma reserva (objeto JSON ou ``dict``), uma lista delas ou um ``DataFrame``, só com o que se sabe no ato
da reserva (``src.schema.COLUNAS_RESERVA``): sem ``is_canceled`` e sem as colunas posteriores. Exemplos em
``exemplos/``. A reserva passa pelo ETL do treino no modo inferência e pelo vigente, carregado como no
notebook 05. A saída traz a probabilidade, a classe e o risco (alto a partir de ``promocao.LIMIAR_CLASSE``,
o mesmo corte da avaliação).

Uso: ``python -m src.prever --json exemplos/reservas_exemplo.json [--manifesto models/modelo_vigente_actions.json]``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import RAIZ, carregar_config, resolver
from src.preprocessamento import limpar_e_engenhar
from src import promocao

EXEMPLO = "exemplos/reservas_exemplo.json"   # usado na demonstração do notebook 05
RISCO_ALTO = "risco alto"
RISCO_BAIXO = "risco baixo"


def ler_reservas(caminho: str | Path) -> list[dict]:
    """Lê um JSON com uma reserva (objeto) ou várias (lista de objetos)."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    reservas = [dados] if isinstance(dados, dict) else dados
    if not isinstance(reservas, list) or not reservas or not all(isinstance(r, dict) for r in reservas):
        raise ValueError(f"{caminho} deve conter um objeto JSON (uma reserva) ou uma lista de objetos")
    return reservas


def carregar_modelo_vigente(cfg: dict, raiz: Path = RAIZ, manifesto: str | Path | None = None):
    """Devolve ``(modelo, entrada_do_vigente)``, como no notebook 05.

    ``manifesto`` permite usar outro manifesto (por exemplo, a cópia do estado do Actions em
    ``models/modelo_vigente_actions.json``); o arquivo do modelo continua sendo procurado em
    ``models/registro/``. Falha com uma mensagem clara se ainda não há vigente.
    """
    if manifesto is None:
        conteudo = promocao.carregar_manifesto(cfg, raiz)
    else:
        conteudo = json.loads(resolver(manifesto, raiz).read_text(encoding="utf-8"))
    modelo, vigente = promocao.carregar_vigente(conteudo, raiz)
    if vigente is None:
        raise FileNotFoundError(
            "Ainda não há modelo vigente: rode um ciclo do pipeline (README, 'Execução local') "
            "ou use o vigente do Actions (README, 'Demonstração de uso')."
        )
    return modelo, vigente


def rotulo_risco(probabilidade: float, limiar: float = promocao.LIMIAR_CLASSE) -> str:
    return RISCO_ALTO if probabilidade >= limiar else RISCO_BAIXO


def prever_reserva(reservas: dict | list[dict] | pd.DataFrame, modelo=None, vigente: dict | None = None,
                   cfg: dict | None = None, raiz: Path = RAIZ) -> pd.DataFrame:
    """Probabilidade de cancelamento de uma ou mais reservas novas. Uma linha por reserva, na ordem da entrada.

    Sem ``modelo``, carrega o vigente (``carregar_modelo_vigente``). Uma reserva que o ETL descartaria
    (sem hóspedes ou com diária negativa) interrompe com erro, em vez de sumir da resposta.
    """
    cfg = cfg or carregar_config()
    if modelo is None:
        modelo, vigente = carregar_modelo_vigente(cfg, raiz)
    vigente = vigente or {}
    if isinstance(reservas, pd.DataFrame):
        entrada = reservas.reset_index(drop=True)
    else:
        entrada = pd.DataFrame([reservas] if isinstance(reservas, dict) else list(reservas))
    if entrada.empty:
        raise ValueError("Nenhuma reserva para prever")

    limpo, rel = limpar_e_engenhar(entrada, cfg, inferencia=True)
    if rel["regras_dominio_removidas"]:
        posicoes = [i + 1 for i in rel["indices_removidos_dominio"]]
        raise ValueError(
            f"Reserva(s) inválida(s) na posição {posicoes}: sem hóspedes ou com diária (adr) negativa. "
            "O pipeline descarta esses registros no ETL, então o modelo não os avalia."
        )

    saida = promocao.prever(modelo, limpo, cfg)
    saida.insert(0, "reserva", range(1, len(saida) + 1))
    saida.insert(1, "hotel", limpo["hotel"].to_numpy())
    saida["limiar"] = promocao.LIMIAR_CLASSE
    saida["risco"] = [rotulo_risco(p) for p in saida["prob_cancelamento"]]
    saida["vigente_cycle_id"] = vigente.get("cycle_id")
    saida["vigente_modelo"] = vigente.get("modelo")
    return saida.reset_index(drop=True)


def salvar_demonstracao(resultado: pd.DataFrame, ctx: dict, raiz: Path = RAIZ) -> Path:
    """Grava ``evidencias/previsoes/demonstracao_<cycle_id>.csv`` (evidência da demonstração no ciclo).

    O nome não começa com ``previsoes_``, então a conferência com o rótulo real (notebook 05) o ignora.
    """
    destino = resolver(promocao.DIR_EVIDENCIAS_PREVISOES, raiz) / f"demonstracao_{ctx['cycle_id']}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.assign(cycle_id=ctx["cycle_id"], data_version=ctx["data_version"]).to_csv(destino, index=False)
    return destino


def _pct(valor: float, casas: int = 1) -> str:
    return f"{valor:.{casas}%}".replace(".", ",")


def descrever(resultado: pd.DataFrame) -> str:
    """Texto para quem opera o hotel: uma linha por reserva, depois o limiar e o modelo usados."""
    linhas = [
        f"Reserva {r.reserva} ({r.hotel}, chegada {r.arrival_date:%d/%m/%Y}): "
        f"chance de cancelamento {_pct(r.prob_cancelamento)} -> {r.risco}"
        for r in resultado.itertuples(index=False)
    ]
    primeira = resultado.iloc[0]
    linhas.append(f"Limiar: {_pct(primeira['limiar'], 0)} (a partir dele, risco alto; o mesmo corte da avaliação do modelo)")
    if primeira["vigente_cycle_id"]:
        linhas.append(f"Modelo vigente: {primeira['vigente_modelo']} do ciclo {primeira['vigente_cycle_id']}")
    return "\n".join(linhas)


def main(argv: list[str] | None = None, raiz: Path = RAIZ) -> int:
    ap = argparse.ArgumentParser(description="Probabilidade de cancelamento de uma reserva nova, com o modelo vigente.")
    ap.add_argument("--json", required=True, help="arquivo com uma reserva (objeto) ou uma lista de reservas")
    ap.add_argument("--manifesto", default=None, help="manifesto do vigente (padrão: models/modelo_vigente.json)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = carregar_config(args.config)
    try:
        modelo, vigente = carregar_modelo_vigente(cfg, raiz, args.manifesto)
        resultado = prever_reserva(ler_reservas(args.json), modelo, vigente, cfg, raiz)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1
    print(descrever(resultado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
