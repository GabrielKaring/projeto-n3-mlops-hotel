"""Divide o CSV original do Kaggle em lotes mensais (uso único, na criação do repositório).

Uso:
    python scripts/preparar_lotes.py caminho/hotel_bookings.csv

Os primeiros ``dados.lotes_iniciais`` meses vão para data/raw (e são registrados no manifesto);
os demais ficam em data/incoming, à espera de serem "liberados" em ciclos futuros.
Cada lote preserva a ordem original das linhas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import carregar_config, resolver  # noqa: E402
from src.ingestao import registrar_lotes_novos  # noqa: E402
from src.schema import COLUNAS_BRUTAS, construir_data_chegada  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="hotel_bookings.csv original")
    args = ap.parse_args()

    cfg = carregar_config()
    dir_raw = resolver(cfg["dados"]["dir_raw"])
    dir_incoming = resolver(cfg["dados"]["dir_incoming"])
    n_iniciais = cfg["dados"]["lotes_iniciais"]

    df = pd.read_csv(args.csv)
    faltando = [c for c in COLUNAS_BRUTAS if c not in df.columns]
    if faltando:
        raise SystemExit(f"CSV sem colunas esperadas: {faltando}")

    periodo = construir_data_chegada(df).dt.to_period("M")
    meses = sorted(periodo.unique())
    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_incoming.mkdir(parents=True, exist_ok=True)

    for i, mes in enumerate(meses):
        destino = (dir_raw if i < n_iniciais else dir_incoming) / f"lote_{mes}.csv.gz"
        # mtime=0 torna o .gz determinístico: o mesmo conteúdo gera sempre o mesmo sha256
        df.loc[periodo == mes, COLUNAS_BRUTAS].to_csv(
            destino, index=False, compression={"method": "gzip", "mtime": 0}
        )

    r = registrar_lotes_novos(dir_raw, resolver(cfg["dados"]["manifesto"]))
    print(f"{len(meses)} lotes gerados: {min(n_iniciais, len(meses))} em data/raw (registrados: {len(r['novos'])}), "
          f"{max(len(meses) - n_iniciais, 0)} em data/incoming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
