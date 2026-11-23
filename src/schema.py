"""Esquema do dataset Hotel Booking Demand e utilitários de data."""
from __future__ import annotations

import pandas as pd

MESES = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

COLUNAS_BRUTAS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
    "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]

ALVO = "is_canceled"

# Colunas que só existem depois da reserva (alterações, check-in ou desfecho). O ETL as descarta antes do modelo,
# então uma reserva nova, no modo inferência, pode chegar sem elas.
COLUNAS_POS_RESERVA = ["assigned_room_type", "booking_changes", "reservation_status", "reservation_status_date"]

# Campos que podem faltar numa reserva nova: o ETL já trata o ausente como "não informado"
# (country -> "UNK", agent -> "SEM_AGENTE", company -> tem_empresa = 0, children -> 0).
COLUNAS_OPCIONAIS_RESERVA = ["country", "agent", "company", "children"]

# O que uma reserva nova precisa trazer no modo inferência: tudo o que se sabe no ato da reserva.
COLUNAS_RESERVA = [
    c for c in COLUNAS_BRUTAS if c not in (ALVO, *COLUNAS_POS_RESERVA, *COLUNAS_OPCIONAIS_RESERVA)
]


def construir_data_chegada(df: pd.DataFrame) -> pd.Series:
    """Monta a data de chegada a partir de ano, mês (nome em inglês) e dia.

    Levanta ValueError se houver mês desconhecido ou data impossível, para que
    o pipeline falhe cedo em vez de propagar dados inválidos.
    """
    mes = df["arrival_date_month"].map(MESES)
    if mes.isna().any():
        invalidos = sorted(df.loc[mes.isna(), "arrival_date_month"].astype(str).unique())
        raise ValueError(f"Meses de chegada desconhecidos: {invalidos}")
    partes = pd.DataFrame(
        {
            "year": df["arrival_date_year"].to_numpy(),
            "month": mes.to_numpy(),
            "day": df["arrival_date_day_of_month"].to_numpy(),
        },
        index=df.index,
    )
    return pd.to_datetime(partes)
