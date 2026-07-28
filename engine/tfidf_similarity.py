# engine/tfidf_similarity.py

"""
FSIS TF-IDF Similarity Metric
------------------------------
Treats each sample as a "document" and each substance as a "term".

TF  = normalised substance percentage (already in feature matrix)
IDF = log((1 + N) / (1 + df)) + 1   (smooth IDF, sklearn convention)
      where df = number of samples in which the substance is present

This down-weights ubiquitous substances (e.g. cocaine in a cocaine dataset)
and up-weights rare substances (e.g. unusual impurities), making rare shared
chemicals a stronger similarity signal.

Similarity is computed as cosine similarity of TF-IDF vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TFIDFSimilarity:
    def __init__(self, feature_matrix: pd.DataFrame):
        if feature_matrix is None or feature_matrix.empty:
            raise ValueError("Feature matrix is empty.")

        self.Xdf = feature_matrix.copy()
        self.samples = self.Xdf.index.astype(str).tolist()
        self.substances = self.Xdf.columns.tolist()
        self.X = self.Xdf.to_numpy(dtype=np.float64, copy=True)

    def _compute_tfidf_matrix(self) -> np.ndarray:
        X = self.X
        N = X.shape[0]

        # TF: raw percentage values (already row-normalised in feature matrix)
        TF = X

        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        df = (X > 0.0).sum(axis=0).astype(np.float64)  # (n_substances,)
        IDF = np.log((1.0 + N) / (1.0 + df)) + 1.0

        tfidf = TF * IDF[np.newaxis, :]
        return tfidf

    def similarity(self) -> pd.DataFrame:
        """
        Returns an (n_samples x n_samples) cosine similarity matrix of
        TF-IDF vectors.
        """
        tfidf = self._compute_tfidf_matrix()

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12
        tfidf_norm = tfidf / norms

        sim = tfidf_norm @ tfidf_norm.T
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        sim = np.round(sim, 6)

        return pd.DataFrame(sim, index=self.samples, columns=self.samples)

    def idf_weights(self) -> pd.Series:
        """Returns IDF weight per substance — useful for inspection/debugging."""
        X = self.X
        N = X.shape[0]
        df = (X > 0.0).sum(axis=0).astype(np.float64)
        IDF = np.log((1.0 + N) / (1.0 + df)) + 1.0
        return pd.Series(IDF, index=self.substances).round(4)
