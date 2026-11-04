"""Gerador de reservas sintéticas com o mesmo esquema do dataset de hotel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import COLUNAS_BRUTAS, MESES

NOMES_MES = {v: k for k, v in MESES.items()}


def gerar_reservas(n: int, ano: int, mes: int, seed: int = 0) -> pd.DataFrame:
    """Gera ``n`` reservas com chegada no mês indicado, já com sujeira típica do dataset."""
    rng = np.random.default_rng(seed + ano * 100 + mes)
    df = pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "is_canceled": rng.integers(0, 2, n),
            "lead_time": rng.integers(0, 400, n),
            "arrival_date_year": ano,
            "arrival_date_month": NOMES_MES[mes],
            "arrival_date_week_number": rng.integers(1, 53, n),
            "arrival_date_day_of_month": rng.integers(1, 28, n),
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 6, n),
            "adults": rng.integers(1, 3, n),
            "children": rng.integers(0, 2, n).astype(float),
            "babies": 0,
            "meal": rng.choice(["BB", "HB", "SC", "Undefined"], n),
            "country": rng.choice(["PRT", "GBR", "FRA", "DEU", "ESP", "BRA"], n),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Groups"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"], n),
            "is_repeated_guest": rng.integers(0, 2, n),
            "previous_cancellations": rng.integers(0, 2, n),
            "previous_bookings_not_canceled": rng.integers(0, 3, n),
            "reserved_room_type": rng.choice(["A", "D", "E"], n),
            "assigned_room_type": rng.choice(["A", "D", "E"], n),
            "booking_changes": rng.integers(0, 3, n),
            "deposit_type": rng.choice(["No Deposit", "Non Refund", "Refundable"], n),
            "agent": rng.choice([np.nan, 9.0, 240.0, 14.0], n),
            "company": rng.choice([np.nan, np.nan, np.nan, 40.0], n),
            "days_in_waiting_list": rng.integers(0, 5, n),
            "customer_type": rng.choice(["Transient", "Contract", "Group"], n),
            "adr": rng.normal(100, 30, n).clip(20, 400).round(2),
            "required_car_parking_spaces": rng.integers(0, 2, n),
            "total_of_special_requests": rng.integers(0, 4, n),
            "reservation_status": rng.choice(["Check-Out", "Canceled"], n),
            "reservation_status_date": "2016-01-01",
        }
    )
    return df[COLUNAS_BRUTAS]
