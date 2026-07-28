# engine/audit.py

"""
FSIS Audit & Run ID (Forensic-grade)
------------------------------------
Creates a deterministic Run ID based on:
- feature matrix hash (index+columns+rounded values)
- normalized weights
- threshold
- engine version
Also includes dataset parsing metadata (e.g., >15 parsed count).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any
import pandas as pd

from .utils import hash_feature_matrix, normalize_weights, round_float
from .version import get_version_info


class FSISAudit:
    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        weights: tuple[float, float, float],
        threshold: float,
        loader_meta: Dict[str, Any] | None = None,
    ):
        if feature_matrix is None or feature_matrix.empty:
            raise ValueError("Feature matrix is empty; cannot audit.")

        self.engine_info = get_version_info()
        self.weights = normalize_weights(weights)
        self.threshold = round_float(threshold, 6)
        self.dataset_hash = hash_feature_matrix(feature_matrix)
        self.loader_meta = loader_meta or {}

        self.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        payload = (
            self.dataset_hash
            + f"|w={self.weights}"
            + f"|t={self.threshold}"
            + f"|v={self.engine_info['engine_version']}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_audit_record(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine_name": self.engine_info["engine_name"],
            "engine_version": self.engine_info["engine_version"],
            "dataset_hash": self.dataset_hash,
            "weights": {
                "cosine": self.weights[0],
                "jaccard": self.weights[1],
                "euclidean": self.weights[2],
            },
            "threshold": self.threshold,
            "loader_meta": self.loader_meta,
        }
