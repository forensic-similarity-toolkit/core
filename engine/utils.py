# engine/utils.py

"""
FSIS Utility Functions
---------------------
Forensic-grade helpers for:
- Feature matrix hashing (Run ID)
- Numeric rounding
- Weight normalization
"""

import hashlib
import pandas as pd
import numpy as np


def round_float(value: float, decimals: int = 6) -> float:
    return round(float(value), decimals)


def normalize_weights(weights: tuple) -> tuple:
    total = float(sum(weights))
    if total == 0:
        raise ValueError("Sum of weights is zero; cannot normalize.")
    return tuple(round_float(w / total, 6) for w in weights)  # type: ignore


def hash_feature_matrix(feature_df: pd.DataFrame) -> str:
    """
    Deterministic SHA256 hash of a feature matrix where:
    - index = Sample IDs
    - columns = substances
    Ensures determinism by sorting index + columns and rounding numeric values.
    """
    if feature_df is None or feature_df.empty:
        raise ValueError("Cannot hash an empty feature matrix.")

    df = feature_df.copy()

    # Sort deterministically
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)

    # Force numeric float64, round
    df = df.apply(pd.to_numeric, errors="raise").astype(np.float64).round(6)

    # Hash CSV with index included (Sample IDs are part of the input contract)
    csv_bytes = df.to_csv(index=True, header=True).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()
