# engine/preprocessor.py

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics_cached import FSISCachedMetrics


class FSISPreprocessor:
    """
    Converts long-format drug sample data into per-category precomputed
    similarity matrices for use with FSISCategoryMetricsStore.

    Input file format (Excel or CSV):
        Long format — one row per sample × substance observation.
        Required columns (names are auto-detected):
            sample      e.g. "Sample Name"
            substance   e.g. "Substance"
            value       e.g. "AREA %"  (may contain "%" and ">" prefixes)
        Optional:
            category    e.g. "Expected Substance"  (used to split into subfolders)
            Any other columns are preserved as sample metadata.

    Output per category subfolder:
        feature_matrix_wide_raw.csv
        feature_matrix_wide_normalized.csv
        cosine_similarity.csv
        jaccard_similarity.csv
        euclidean_similarity.csv
        joint_similarity.csv
        long_format_cleaned.csv
        sample_metadata.csv
        precompute_metadata.json
    """

    _SAMPLE_ALIASES = {
        "sample name", "sample id", "sample", "id", "samplename", "sampleid",
    }
    _SUBSTANCE_ALIASES = {
        "substance", "chemical", "compound", "analyte", "drug substance",
    }
    _VALUE_ALIASES = {
        "area %", "area%", "substance percentage", "percentage", "pct",
        "value", "concentration", "areapct", "areamatch%", "area_pct",
    }
    _CATEGORY_ALIASES = {
        "expected substance", "expect substance", "expectedsubstance",
        "expectsubstance", "category", "drug category", "drug",
    }

    # ------------------------------------------------------------------ #
    # Column detection                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm(c: str) -> str:
        return str(c).strip().lower().replace("_", " ").replace("-", " ")

    def detect_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        """
        Return best guesses for column roles.
        Keys: 'sample', 'substance', 'value', 'category'
        Values: matching column name from df, or None.
        """
        normed = {self._norm(c): c for c in df.columns}
        result: dict[str, str | None] = {}
        for role, aliases in (
            ("sample", self._SAMPLE_ALIASES),
            ("substance", self._SUBSTANCE_ALIASES),
            ("value", self._VALUE_ALIASES),
            ("category", self._CATEGORY_ALIASES),
        ):
            found = None
            for alias in aliases:
                if alias in normed:
                    found = normed[alias]
                    break
            result[role] = found
        return result

    # ------------------------------------------------------------------ #
    # File loading                                                          #
    # ------------------------------------------------------------------ #

    def load_file(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
        # Strip BOM and whitespace from headers
        df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Category listing                                                      #
    # ------------------------------------------------------------------ #

    def list_categories(self, df: pd.DataFrame, category_col: str) -> list[str]:
        vals = df[category_col].astype(str).str.strip()
        return sorted((v for v in vals.unique() if v and v.lower() != "nan"), key=str.lower)

    # ------------------------------------------------------------------ #
    # Value cleaning                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_value_series(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .pipe(lambda x: pd.to_numeric(x, errors="coerce"))
            .fillna(0.0)
            .clip(lower=0.0)
        )

    # ------------------------------------------------------------------ #
    # Core processing                                                       #
    # ------------------------------------------------------------------ #

    def process_category(
        self,
        df: pd.DataFrame,
        category_name: str,
        sample_col: str,
        substance_col: str,
        value_col: str,
        category_col: str | None,
        output_dir: Path,
        source_file: str = "",
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Process one category and write all output files to output_dir.
        Returns the precompute metadata dict.
        Raises ValueError / IOError on problems.
        """

        def log(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Filter by category ----
        if category_col and category_name:
            mask = df[category_col].astype(str).str.strip().str.lower() == category_name.lower()
            cat_df = df[mask].copy()
        else:
            cat_df = df.copy()

        if cat_df.empty:
            raise ValueError(f"No rows found for category: {category_name!r}")

        # ---- Clean value column ----
        cat_df = cat_df.copy()
        cat_df[value_col] = self._clean_value_series(cat_df[value_col])

        # ---- Long format cleaned ----
        cat_df.reset_index(drop=True).to_csv(output_dir / "long_format_cleaned.csv")
        log(f"  Saved long_format_cleaned.csv  ({len(cat_df)} rows)")

        # ---- Sample metadata ----
        # Every column except the per-row substance/value columns
        meta_cols = [c for c in cat_df.columns if c not in (substance_col, value_col)]
        meta_df = (
            cat_df[meta_cols]
            .drop_duplicates(subset=[sample_col])
            .reset_index(drop=True)
        )
        meta_df.to_csv(output_dir / "sample_metadata.csv", index=False)
        log(f"  Saved sample_metadata.csv  ({len(meta_df)} samples)")

        # ---- Wide pivot ----
        feature_matrix: pd.DataFrame = cat_df.pivot_table(
            index=sample_col,
            columns=substance_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        feature_matrix = feature_matrix.sort_index().sort_index(axis=1)
        feature_matrix = feature_matrix.astype(np.float64).round(6)
        feature_matrix.index.name = sample_col

        n_samples = int(feature_matrix.shape[0])
        n_substances = int(feature_matrix.shape[1])

        if n_samples < 2:
            raise ValueError(
                f"Category {category_name!r} has only {n_samples} sample(s). "
                "At least 2 are required to compute similarity matrices."
            )

        feature_matrix.to_csv(output_dir / "feature_matrix_wide_raw.csv")
        log(f"  Saved feature_matrix_wide_raw.csv  ({n_samples} × {n_substances})")

        # ---- Row-normalise ----
        row_sums = feature_matrix.sum(axis=1).replace(0, np.nan)
        feature_matrix_norm = feature_matrix.div(row_sums, axis=0).fillna(0.0).round(6)
        feature_matrix_norm.to_csv(output_dir / "feature_matrix_wide_normalized.csv")
        log(f"  Saved feature_matrix_wide_normalized.csv")

        # ---- Similarity matrices (via FSISCachedMetrics) ----
        # Pass normalize_rows=False — we have already normalised above.
        log(f"  Computing similarity matrices...")
        cached = FSISCachedMetrics(feature_matrix_norm, normalize_rows=False)
        cos_df, jac_df, euc_df = cached.compute_base_metrics()

        cos_df.round(6).to_csv(output_dir / "cosine_similarity.csv")
        jac_df.round(6).to_csv(output_dir / "jaccard_similarity.csv")
        euc_df.round(6).to_csv(output_dir / "euclidean_similarity.csv")
        log(f"  Saved cosine, jaccard, euclidean similarity matrices")

        # ---- Default joint similarity (equal weights) ----
        joint_df = cached.joint_similarity((1 / 3, 1 / 3, 1 / 3))
        joint_df.round(6).to_csv(output_dir / "joint_similarity.csv")
        log(f"  Saved joint_similarity.csv")

        # ---- Metadata JSON ----
        metadata: dict[str, Any] = {
            "source_file": source_file,
            "category_name": category_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "required_columns_used": {
                "sample": sample_col,
                "substance": substance_col,
                "value": value_col,
                "category": category_col,
            },
            "n_long_rows_cleaned": int(len(cat_df)),
            "n_samples": n_samples,
            "n_substances": n_substances,
            "normalize_rows": True,
            "presence_eps": 0.0,
            "round_decimals": 6,
            "duplicate_aggregation": "sum",
            "files": {
                "cleaned_long": "long_format_cleaned.csv",
                "sample_metadata": "sample_metadata.csv",
                "feature_matrix_wide_raw": "feature_matrix_wide_raw.csv",
                "feature_matrix_used": "feature_matrix_wide_normalized.csv",
                "cosine_similarity": "cosine_similarity.csv",
                "jaccard_similarity": "jaccard_similarity.csv",
                "euclidean_similarity": "euclidean_similarity.csv",
                "joint_similarity": "joint_similarity.csv",
            },
        }
        with open(output_dir / "precompute_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        log(f"  Saved precompute_metadata.json")

        return metadata
