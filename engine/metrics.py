# engine/metrics.py

"""
FSIS Similarity Metrics (Real-time)
-----------------------------------
Computes:
- Cosine similarity
- Jaccard similarity (binary presence/absence)
- Euclidean distance -> normalized similarity
- Geometric joint similarity from the three metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import normalize_weights


class FSISMetrics:
    def __init__(self, feature_matrix: pd.DataFrame):
        if feature_matrix is None or feature_matrix.empty:
            raise ValueError("Feature matrix is empty.")

        self.Xdf = feature_matrix.copy()
        self.samples = self.Xdf.index.astype(str).tolist()

        # stable numeric matrix
        self.X = self.Xdf.to_numpy(dtype=np.float64, copy=True)

    # ---------------- Base metrics ----------------

    def cosine_similarity(self) -> pd.DataFrame:
        X = self.X

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12

        Xn = X / norms
        sim = Xn @ Xn.T

        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)

        sim = np.round(sim, 6)
        return pd.DataFrame(sim, index=self.samples, columns=self.samples)

    def jaccard_similarity(self) -> pd.DataFrame:
        Xb = (self.X > 0.0).astype(np.int32)

        inter = Xb @ Xb.T
        sums = Xb.sum(axis=1).reshape(-1, 1)
        union = sums + sums.T - inter

        union = np.where(union == 0, 1, union)
        sim = inter / union

        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)

        sim = np.round(sim, 6)
        return pd.DataFrame(sim, index=self.samples, columns=self.samples)

    def pearson_similarity(self) -> pd.DataFrame:
        """
        Pearson correlation as similarity.
        Equivalent to cosine similarity on mean-centred rows.
        Negative correlations are clipped to 0 (anti-patterns are not similar).
        """
        X = self.X
        Xc = X - X.mean(axis=1, keepdims=True)

        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12

        Xn = Xc / norms
        sim = Xn @ Xn.T

        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)

        sim = np.round(sim, 6)
        return pd.DataFrame(sim, index=self.samples, columns=self.samples)

    def euclidean_similarity(self) -> pd.DataFrame:
        """
        Euclidean similarity:
            d(i,j) = sqrt(||xi||^2 + ||xj||^2 - 2 xi·xj)
            sim(i,j) = 1 / (1 + d(i,j))
        """
        X = self.X

        sq = np.sum(X * X, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)

        d = np.sqrt(d2)
        sim = 1.0 / (1.0 + d)

        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)

        sim = np.round(sim, 6)
        return pd.DataFrame(sim, index=self.samples, columns=self.samples)

    # ---------------- Joint similarity ----------------

    @staticmethod
    def _power_component(mat: np.ndarray, w: float) -> np.ndarray:
        """
        If weight is zero, treat the component as neutral (all ones).
        """
        if w == 0.0:
            return np.ones_like(mat, dtype=np.float64)
        return np.power(mat, w)

    def joint_similarity(self, weights: tuple) -> pd.DataFrame:
        """
        Geometric fusion:
            JS = Cos^w1 * Jac^w2 * Euc^w3 * Pea^w4
        Accepts a 3-tuple (cosine, jaccard, euclidean) or 4-tuple (+ pearson).
        """
        w = normalize_weights(weights)
        w_cos, w_jac, w_euc = w[0], w[1], w[2]
        w_pea = w[3] if len(w) > 3 else 0.0

        cos = self.cosine_similarity().to_numpy(dtype=np.float64)
        jac = self.jaccard_similarity().to_numpy(dtype=np.float64)
        euc = self.euclidean_similarity().to_numpy(dtype=np.float64)

        joint = (
            self._power_component(cos, w_cos)
            * self._power_component(jac, w_jac)
            * self._power_component(euc, w_euc)
            * self._power_component(self.pearson_similarity().to_numpy(dtype=np.float64), w_pea)
        )

        joint = np.clip(joint, 0.0, 1.0)
        np.fill_diagonal(joint, 1.0)

        joint = np.round(joint, 6)
        return pd.DataFrame(joint, index=self.samples, columns=self.samples)
