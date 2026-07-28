# engine/data_loader.py

"""
FSIS Data Loader
----------------
Loads and validates FSIS datasets and constructs a deterministic feature matrix.
"""

from __future__ import annotations

import re
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


# Matches values like ">15"
_GT_RE = re.compile(r"^\s*>\s*([0-9]*\.?[0-9]+)\s*$")


def _norm_col(c: str) -> str:
    """
    Normalize column headers:
    - remove BOM
    - lowercase
    - underscores → space
    - collapse spaces
    """
    s = str(c).replace("\ufeff", "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


class FSISDataLoader:
    def __init__(
        self,
        filepath: str,
        sample_id_col: str | None = None,
        unknown_col: str = "Unknown Substances",
    ):
        self.filepath = filepath
        self.sample_id_col = sample_id_col
        self.unknown_col = unknown_col

        self.df_raw: pd.DataFrame | None = None
        self.feature_matrix: pd.DataFrame | None = None
        self.meta: Dict[str, Any] = {}

    # ---------------- Parsing ----------------

    @staticmethod
    def _parse_cell(val) -> tuple[float, bool, bool]:
        """
        Returns:
        (value, gt_flag, blank_flag)
        """

        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0, False, True

        if isinstance(val, str):
            s = val.strip()

            if s == "":
                return 0.0, False, True

            m = _GT_RE.match(s)
            if m:
                return float(m.group(1)), True, False

            try:
                return float(s.replace(",", "")), False, False
            except ValueError:
                raise ValueError(f"Non-numeric value: '{val}'")

        try:
            return float(val), False, False
        except Exception:
            raise ValueError(f"Non-numeric value: '{val}'")

    # ---------------- Unknown column detection ----------------

    def _detect_unknown_col(self, df: pd.DataFrame) -> str | None:
        want = _norm_col(self.unknown_col)

        norm_map = {_norm_col(c): c for c in df.columns}

        if want in norm_map:
            return norm_map[want]

        # fallback matches
        for key in ("unknown substances", "unknowns", "unknown substance"):
            if key in norm_map:
                return norm_map[key]

        return None

    # ---------------- Main loader ----------------

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

        # Read dataset
        df = pd.read_csv(self.filepath, sep=None, engine="python")

        if df.empty:
            raise ValueError(f"Dataset is empty: {self.filepath}")

        # Clean headers
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

        # Identify sample column
        sample_col = self.sample_id_col or df.columns[0]

        if sample_col not in df.columns:
            raise ValueError(f"Sample ID column not found: {sample_col}")

        df[sample_col] = df[sample_col].astype(str).str.strip()

        # Duplicate check
        if df[sample_col].duplicated().any():
            dups = df.loc[df[sample_col].duplicated(), sample_col].tolist()
            raise ValueError(f"Duplicate Sample IDs found: {dups[:20]}")

        # Detect unknown column
        unknown_actual = self._detect_unknown_col(df)
        unknown_present = unknown_actual is not None

        # Feature columns
        exclude = {sample_col}
        if unknown_present:
            exclude.add(unknown_actual)

        feature_cols = [c for c in df.columns if c not in exclude]

        if not feature_cols:
            raise ValueError("No feature columns found.")

        # Parse features
        parsed = {}
        gt_count = 0
        blank_count = 0

        for col in feature_cols:
            col_vals = []

            for v in df[col].tolist():
                x, gt, blank = self._parse_cell(v)

                if x < 0:
                    raise ValueError(f"Negative value in '{col}': {v}")

                gt_count += int(gt)
                blank_count += int(blank)

                col_vals.append(x)

            parsed[col] = col_vals

        feature_df = pd.DataFrame(parsed, index=df[sample_col].tolist())

        # Deterministic ordering
        feature_df.index.name = sample_col
        feature_df = feature_df.sort_index()
        feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)

        # Numeric stability
        feature_df = feature_df.astype(np.float64).round(6)

        # Store
        self.df_raw = df
        self.feature_matrix = feature_df

        self.meta = {
            "sample_id_col": sample_col,
            "unknown_col_present": bool(unknown_present),
            "unknown_col_name": unknown_actual if unknown_present else None,
            "gt_parsed_count": int(gt_count),
            "blank_as_zero_count": int(blank_count),
            "n_samples": int(feature_df.shape[0]),
            "n_features": int(feature_df.shape[1]),
        }

        return feature_df, df, self.meta


if __name__ == "__main__":
    loader = FSISDataLoader("data/sample_dataset.csv")
    X, raw, meta = loader.load()

    print(meta)
    print(X.head())
