"""Caricamento dei dataset e gruppi di colonne condivisi da tutti i notebook."""

from __future__ import annotations

import pandas as pd

from .paths import DATA_DIR, data_file

# Colonne che identificano il flusso (chi parla con chi, e quando).
# Non descrivono il comportamento del traffico: un modello che le usa
# impara "chi" invece di "come".
ID_COLS = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port"]

# Colonne categoriche che possono diventare feature, se codificate.
CATEGORICAL_COLS = ["protocol"]

# Colonne di etichetta e metadati aggiunte da scripts/prepare_data.py.
LABEL_COLS = ["label", "label_binary", "family", "day"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def load_static(columns: list[str] | None = None) -> pd.DataFrame:
    """Campione per la Lezione 1 (NIDS statici), ordinato per timestamp."""
    return pd.read_parquet(data_file("static.parquet"), columns=columns)


def load_stream(columns: list[str] | None = None) -> pd.DataFrame:
    """Campione per la Lezione 2 (drift e apprendimento online), ordinato per timestamp."""
    return pd.read_parquet(data_file("stream.parquet"), columns=columns)


def load_class_counts() -> pd.DataFrame:
    """Conteggi per classe: dataset originale, campioni, flussi da 1 pacchetto."""
    return pd.read_csv(DATA_DIR / "class_counts.csv", index_col="label", encoding="utf-8")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Le colonne numeriche che descrivono il traffico: tutto tranne identificativi,
    colonne categoriche ed etichette."""
    excluded = set(ID_COLS) | set(CATEGORICAL_COLS) | set(LABEL_COLS)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
