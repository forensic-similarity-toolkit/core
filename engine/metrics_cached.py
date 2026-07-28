# engine/metrics_cached.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .utils import normalize_weights


class FSISCachedMetrics:
    """
    Forensic-grade cached similarity metrics.

    Key update (v1.0):
    - Row-normalize features to proportions by default (normalize_rows=True).
      This prevents Euclidean similarity from collapsing due to magnitude scaling.

    Jaccard:
    - Uses presence_eps threshold (default 0.0 -> present if >0).
    """

    def __init__(self, feature_matrix: pd.DataFrame, normalize_rows: bool = True, presence_eps: float = 0.0):
        if feature_matrix is None or feature_matrix.empty:
            raise ValueError("Feature matrix is empty.")

        self.normalize_rows = bool(normalize_rows)
        self.presence_eps = float(presence_eps)

        # Store a normalized (or original) DF used for actual computations
        df = feature_matrix.copy()
        df.index = df.index.astype(str)

        if self.normalize_rows:
            row_sum = df.sum(axis=1)
            row_sum = row_sum.replace(0, 1.0)
            df = df.div(row_sum, axis=0)

        self.Xdf = df
        self.samples = self.Xdf.index.astype(str).tolist()

        # ✅ INSERTED: mapping from sample id to row index for fast lookup
        self._idx = {sid: i for i, sid in enumerate(self.samples)}

        self.X = self.Xdf.to_numpy(dtype=np.float64, copy=True)

        self._cos: Optional[np.ndarray] = None
        self._jac: Optional[np.ndarray] = None
        self._euc: Optional[np.ndarray] = None
        self._pea: Optional[np.ndarray] = None

    # ✅ INSERTED: pair breakdown method (cosine/jaccard/euclidean for a pair)
    def get_pair_breakdown(self, a: str, b: str) -> dict:
        """
        Returns the four base similarity metrics for a selected pair.
        Assumes both 'a' and 'b' are valid sample IDs.
        """
        if self._cos is None or self._jac is None or self._euc is None or self._pea is None:
            self.compute_base_metrics()

        ia = self._idx[a]
        ib = self._idx[b]
        return {
            "cosine": float(self._cos[ia, ib]),
            "jaccard": float(self._jac[ia, ib]),
            "euclidean": float(self._euc[ia, ib]),
            "pearson": float(self._pea[ia, ib]),
        }

    def compute_base_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._cos is None:
            self._cos = self._cosine_np()
        if self._jac is None:
            self._jac = self._jaccard_np()
        if self._euc is None:
            self._euc = self._euclidean_sim_np()
        if self._pea is None:
            self._pea = self._pearson_np()

        cos_df = pd.DataFrame(self._cos, index=self.samples, columns=self.samples)
        jac_df = pd.DataFrame(self._jac, index=self.samples, columns=self.samples)
        euc_df = pd.DataFrame(self._euc, index=self.samples, columns=self.samples)
        pea_df = pd.DataFrame(self._pea, index=self.samples, columns=self.samples)
        return cos_df, jac_df, euc_df, pea_df

    def joint_similarity(self, weights: tuple) -> pd.DataFrame:
        w = normalize_weights(weights)
        w_cos, w_jac, w_euc = w[0], w[1], w[2]
        w_pea = w[3] if len(w) > 3 else 0.0

        if self._cos is None:
            self._cos = self._cosine_np()
        if self._jac is None:
            self._jac = self._jaccard_np()
        if self._euc is None:
            self._euc = self._euclidean_sim_np()
        if self._pea is None:
            self._pea = self._pearson_np()

        joint = (
            self._pow_component(self._cos, w_cos)
            * self._pow_component(self._jac, w_jac)
            * self._pow_component(self._euc, w_euc)
            * self._pow_component(self._pea, w_pea)
        )
        joint = np.clip(joint, 0.0, 1.0)
        joint = np.round(joint, 6)
        return pd.DataFrame(joint, index=self.samples, columns=self.samples)

    @staticmethod
    def _pow_component(mat: np.ndarray, w: float) -> np.ndarray:
        if w == 0.0:
            return np.ones_like(mat, dtype=np.float64)
        return np.power(mat, w)

    def _cosine_np(self) -> np.ndarray:
        X = self.X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12
        Xn = X / norms
        sim = Xn @ Xn.T
        sim = np.clip(sim, 0.0, 1.0)
        # Force perfect diagonal
        np.fill_diagonal(sim, 1.0)
        return np.round(sim, 6)

    def _jaccard_np(self) -> np.ndarray:
        # Presence threshold for Jaccard
        Xb = (self.X > self.presence_eps).astype(np.int32)

        inter = Xb @ Xb.T
        sums = Xb.sum(axis=1).reshape(-1, 1)
        union = sums + sums.T - inter

        union = np.where(union == 0, 1, union)
        sim = inter / union
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return np.round(sim, 6)

    def _pearson_np(self) -> np.ndarray:
        """
        Pearson correlation as similarity (mean-centred cosine).
        Negative correlations clipped to 0; anti-patterns are not similar.
        """
        X = self.X
        Xc = X - X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12
        Xn = Xc / norms
        sim = Xn @ Xn.T
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return np.round(sim, 6)

    def _euclidean_sim_np(self) -> np.ndarray:
        """
        Euclidean similarity: 1 / (1 + distance)
        With normalize_rows=True, distance is computed on proportions -> stable and meaningful.
        """
        X = self.X
        sq = np.sum(X * X, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2)

        sim = 1.0 / (1.0 + d)
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        return np.round(sim, 6)
