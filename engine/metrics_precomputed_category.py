from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FSISCategoryMetricsStore:
    """
    Loads precomputed similarity matrices for ONE Expected Substance category.

    Matches your current folder structure:

        precomputed_metrics/
            Cocaine/
                cosine_similarity.csv
                jaccard_similarity.csv
                euclidean_similarity.csv
                feature_matrix_wide_raw.csv
                feature_matrix_wide_normalized.csv
                long_format_cleaned.csv
                sample_metadata.csv
                precompute_metadata.json

    This class is also compatible with PairMixin because it provides:
        - Xdf
        - get_pair_breakdown(...)
    """

    def __init__(self, base_dir: str | Path = "precomputed_metrics") -> None:
        self.base_dir = Path(base_dir)

        self.category_name: str | None = None
        self.category_dir: Path | None = None
        self.metadata: dict[str, Any] = {}

        self.feature_matrix_df: pd.DataFrame | None = None
        self.feature_matrix_raw_df: pd.DataFrame | None = None
        self.long_format_df: pd.DataFrame | None = None
        self.sample_metadata_df: pd.DataFrame | None = None

        self.cosine_df: pd.DataFrame | None = None
        self.jaccard_df: pd.DataFrame | None = None
        self.euclidean_df: pd.DataFrame | None = None
        self.joint_default_df: pd.DataFrame | None = None
        self._pearson_df: pd.DataFrame | None = None  # computed on demand from feature matrix

    @property
    def Xdf(self) -> pd.DataFrame | None:
        """
        Compatibility alias for existing FSIS pair-detail logic.
        """
        return self.feature_matrix_df

    def list_categories(self) -> list[str]:
        if not self.base_dir.exists():
            return []

        required = {
            "cosine_similarity.csv",
            "jaccard_similarity.csv",
            "euclidean_similarity.csv",
            "feature_matrix_wide_normalized.csv",
        }

        categories: list[str] = []
        for p in sorted(self.base_dir.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_dir():
                continue
            existing = {child.name for child in p.iterdir() if child.is_file()}
            if required.issubset(existing):
                categories.append(p.name)
        return categories

    def has_category(self, category_name: str) -> bool:
        return (self.base_dir / category_name).is_dir()

    def load_category(self, category_name: str) -> None:
        category_dir = self.base_dir / category_name

        if not category_dir.exists() or not category_dir.is_dir():
            raise FileNotFoundError(f"Category folder not found: {category_dir}")

        metadata = self._read_json_if_exists(category_dir / "precompute_metadata.json")

        cosine_df = self._read_similarity_matrix(category_dir / "cosine_similarity.csv")
        jaccard_df = self._read_similarity_matrix(category_dir / "jaccard_similarity.csv")
        euclidean_df = self._read_similarity_matrix(category_dir / "euclidean_similarity.csv")
        joint_default_df = self._read_similarity_matrix_if_exists(category_dir / "joint_similarity.csv")
        if joint_default_df is None:
            joint_default_df = self._read_similarity_matrix_if_exists(category_dir / "joint_similarity_default.csv")

        feature_matrix_df = self._read_feature_matrix_if_exists(
            category_dir / "feature_matrix_wide_normalized.csv"
        )

        feature_matrix_raw_df = self._read_feature_matrix_if_exists(
            category_dir / "feature_matrix_wide_raw.csv"
        )

        long_format_df = self._read_table_if_exists(
            category_dir / "long_format_cleaned.csv"
        )

        sample_metadata_df = self._read_table_if_exists(
            category_dir / "sample_metadata.csv"
        )

        self._validate_loaded_data(
            cosine_df=cosine_df,
            jaccard_df=jaccard_df,
            euclidean_df=euclidean_df,
            feature_matrix_df=feature_matrix_df,
        )

        # The saved joint matrix is optional. It must match the exact same
        # sample IDs as the base metric matrices. If it does not, ignore it and
        # safely fall back to recomputing the equal-weight default joint matrix
        # from cosine/Jaccard/Euclidean. This prevents stale/full-dataset joint
        # files from being used inside a category folder.
        if joint_default_df is not None:
            if (
                not joint_default_df.index.equals(cosine_df.index)
                or not joint_default_df.columns.equals(cosine_df.columns)
            ):
                joint_default_df = None

        self.category_name = category_name
        self.category_dir = category_dir
        self.metadata = metadata

        self.cosine_df = cosine_df
        self.jaccard_df = jaccard_df
        self.euclidean_df = euclidean_df
        self.joint_default_df = joint_default_df

        self.feature_matrix_df = feature_matrix_df
        self.feature_matrix_raw_df = feature_matrix_raw_df
        self.long_format_df = long_format_df
        self.sample_metadata_df = sample_metadata_df

    def clear(self) -> None:
        self.category_name = None
        self.category_dir = None
        self.metadata = {}

        self.feature_matrix_df = None
        self.feature_matrix_raw_df = None
        self.long_format_df = None
        self.sample_metadata_df = None

        self.cosine_df = None
        self.jaccard_df = None
        self.euclidean_df = None
        self.joint_default_df = None
        self._pearson_df = None

    def is_loaded(self) -> bool:
        return (
            self.category_name is not None
            and self.cosine_df is not None
            and self.jaccard_df is not None
            and self.euclidean_df is not None
        )

    def sample_ids(self) -> list[str]:
        self._require_loaded()
        assert self.cosine_df is not None
        return list(self.cosine_df.index)

    def n_samples(self) -> int:
        self._require_loaded()
        assert self.cosine_df is not None
        return int(self.cosine_df.shape[0])

    def n_features(self) -> int:
        if self.feature_matrix_df is not None:
            return int(self.feature_matrix_df.shape[1])

        value = self.metadata.get("n_features", 0)
        try:
            return int(value)
        except Exception:
            return 0

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "precomputed_category",
            "category_name": self.category_name,
            "category_dir": str(self.category_dir) if self.category_dir else "",
            "n_samples": self.n_samples() if self.is_loaded() else 0,
            "n_features": self.n_features() if self.is_loaded() else 0,
            "has_feature_matrix": self.feature_matrix_df is not None,
            "has_raw_feature_matrix": self.feature_matrix_raw_df is not None,
            "has_long_format": self.long_format_df is not None,
            "has_sample_metadata": self.sample_metadata_df is not None,
            "has_default_joint_similarity": self.joint_default_df is not None,
            "metadata": self.metadata,
        }

    def get_pair_breakdown(self, sample_a: str, sample_b: str) -> dict[str, float]:
        """
        Compatibility method for PairMixin.

        Returns base metric values for a sample pair.
        """
        self._require_loaded()

        assert self.cosine_df is not None
        assert self.jaccard_df is not None
        assert self.euclidean_df is not None

        sample_a = str(sample_a)
        sample_b = str(sample_b)

        if sample_a not in self.cosine_df.index:
            raise KeyError(f"Sample not found in cosine matrix: {sample_a}")

        if sample_b not in self.cosine_df.columns:
            raise KeyError(f"Sample not found in cosine matrix: {sample_b}")

        if self._pearson_df is None:
            self._pearson_df = self._compute_pearson()
        pearson_val = 0.0
        if (
            self._pearson_df is not None
            and sample_a in self._pearson_df.index
            and sample_b in self._pearson_df.columns
        ):
            pearson_val = float(self._pearson_df.loc[sample_a, sample_b])

        return {
            "cosine": float(self.cosine_df.loc[sample_a, sample_b]),
            "jaccard": float(self.jaccard_df.loc[sample_a, sample_b]),
            "euclidean": float(self.euclidean_df.loc[sample_a, sample_b]),
            "pearson": pearson_val,
        }


    def default_joint_similarity(self) -> pd.DataFrame:
        """
        Return the prepared default joint matrix when available.

        If a category has no saved joint_similarity.csv, fall back to equal-weight
        geometric joint similarity so one-sample/small folders still work.
        """
        self._require_loaded()
        if self.joint_default_df is not None:
            return self.joint_default_df.copy().round(6)
        return self.joint_similarity(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    def joint_similarity(
        self,
        wcos: float,
        wjac: float,
        weuc: float,
        wpea: float = 0.0,
    ) -> pd.DataFrame:
        self._require_loaded()

        assert self.cosine_df is not None
        assert self.jaccard_df is not None
        assert self.euclidean_df is not None

        wcos_n, wjac_n, weuc_n, wpea_n = self._normalize_weights(wcos, wjac, weuc, wpea)

        cosine_values = np.clip(self.cosine_df.to_numpy(dtype=float), 0.0, 1.0)
        jaccard_values = np.clip(self.jaccard_df.to_numpy(dtype=float), 0.0, 1.0)
        euclidean_values = np.clip(self.euclidean_df.to_numpy(dtype=float), 0.0, 1.0)

        joint_values = (
            np.power(cosine_values, wcos_n)
            * np.power(jaccard_values, wjac_n)
            * np.power(euclidean_values, weuc_n)
        )

        if wpea_n > 0.0:
            if self._pearson_df is None:
                self._pearson_df = self._compute_pearson()
            if self._pearson_df is not None:
                pearson_values = np.clip(self._pearson_df.to_numpy(dtype=float), 0.0, 1.0)
                joint_values = joint_values * np.power(pearson_values, wpea_n)

        return pd.DataFrame(
            joint_values,
            index=self.cosine_df.index.copy(),
            columns=self.cosine_df.columns.copy(),
        ).round(6)

    def build_edges(
        self,
        joint_df: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        threshold = float(threshold)

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        if joint_df is None or joint_df.empty:
            return pd.DataFrame(
                columns=["Sample_A", "Sample_B", "joint_similarity", "category"]
            )

        labels = list(joint_df.index)
        rows: list[dict[str, Any]] = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                score = float(joint_df.iat[i, j])
                score_rounded = round(score, 6)

                # Zero-similarity pairs are not forensic links and should remain
                # isolated nodes, even when threshold is set to 0.
                if score_rounded > 0.0 and score_rounded >= threshold:
                    rows.append(
                        {
                            "Sample_A": labels[i],
                            "Sample_B": labels[j],
                            "joint_similarity": score_rounded,
                            "category": self.category_name or "",
                        }
                    )

        return pd.DataFrame(
            rows,
            columns=["Sample_A", "Sample_B", "joint_similarity", "category"],
        )

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise RuntimeError("No precomputed Expected Substance category is loaded.")

    def _read_json_if_exists(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"JSON file must contain an object: {path}")

        return data

    def _read_similarity_matrix(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed similarity matrix: {path}")

        df = pd.read_csv(path, index_col=0)

        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        df = df.sort_index().sort_index(axis=1)
        df = df.apply(pd.to_numeric, errors="raise")
        df = df.astype(float).round(6)

        return df

    def _read_similarity_matrix_if_exists(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return self._read_similarity_matrix(path)

    def _read_feature_matrix_if_exists(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None

        df = pd.read_csv(path, index_col=0)

        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        df = df.sort_index().sort_index(axis=1)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df = df.astype(float).round(6)

        return df

    def _read_table_if_exists(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None

        return pd.read_csv(path)

    def _validate_loaded_data(
        self,
        cosine_df: pd.DataFrame,
        jaccard_df: pd.DataFrame,
        euclidean_df: pd.DataFrame,
        feature_matrix_df: pd.DataFrame | None,
    ) -> None:
        self._validate_square_matrix(cosine_df, "Cosine")
        self._validate_square_matrix(jaccard_df, "Jaccard")
        self._validate_square_matrix(euclidean_df, "Euclidean")

        if not cosine_df.index.equals(jaccard_df.index):
            raise ValueError("Cosine and Jaccard sample IDs do not match.")

        if not cosine_df.index.equals(euclidean_df.index):
            raise ValueError("Cosine and Euclidean sample IDs do not match.")

        if not cosine_df.columns.equals(jaccard_df.columns):
            raise ValueError("Cosine and Jaccard matrix columns do not match.")

        if not cosine_df.columns.equals(euclidean_df.columns):
            raise ValueError("Cosine and Euclidean matrix columns do not match.")

        if feature_matrix_df is not None:
            if not cosine_df.index.equals(feature_matrix_df.index):
                raise ValueError(
                    "Feature matrix sample IDs do not match similarity matrix sample IDs."
                )

    def _validate_square_matrix(self, df: pd.DataFrame, name: str) -> None:
        if df.shape[0] != df.shape[1]:
            raise ValueError(f"{name} similarity matrix is not square.")

        if not df.index.equals(df.columns):
            raise ValueError(f"{name} similarity matrix index and columns do not match.")

    def _compute_pearson(self) -> pd.DataFrame | None:
        """Compute Pearson (mean-centred cosine) similarity from the feature matrix."""
        if self.feature_matrix_df is None or self.feature_matrix_df.empty:
            return None

        X = self.feature_matrix_df.to_numpy(dtype=float)
        Xc = X - X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12
        Xn = Xc / norms
        sim = Xn @ Xn.T
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        sim = np.round(sim, 6)
        return pd.DataFrame(
            sim,
            index=self.feature_matrix_df.index.copy(),
            columns=self.feature_matrix_df.index.copy(),
        )

    def _normalize_weights(
        self,
        wcos: float,
        wjac: float,
        weuc: float,
        wpea: float = 0.0,
    ) -> tuple[float, float, float, float]:
        weights = np.array([wcos, wjac, weuc, wpea], dtype=float)

        if np.any(weights < 0):
            raise ValueError("Similarity weights must be non-negative.")

        total = float(weights.sum())

        if total <= 0:
            raise ValueError("At least one similarity weight must be greater than zero.")

        weights = weights / total

        return float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])