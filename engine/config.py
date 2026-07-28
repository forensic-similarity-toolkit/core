# engine/config.py

"""
FSIS Engine Configuration
-------------------------

Defines a frozen configuration object for the FSIS engine.
Ensures deterministic runs and forensic traceability.
"""

from dataclasses import dataclass, field
from typing import Tuple

from .version import get_version_info


@dataclass(frozen=True)
class FSISConfig:
    """
    Immutable configuration object for FSIS engine.
    All runs use this configuration for reproducibility.
    """

    weight_cosine: float = 0.33
    weight_jaccard: float = 0.33
    weight_euclidean: float = 0.34
    weight_pearson: float = 0.0

    threshold: float = 0.5  # Joint similarity threshold

    engine_version: str = field(
        default_factory=lambda: get_version_info()["engine_version"]
    )

    # ---------------- Validation ----------------

    def __post_init__(self):
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")

        weights = [self.weight_cosine, self.weight_jaccard, self.weight_euclidean, self.weight_pearson]

        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        if sum(weights) == 0:
            raise ValueError("At least one weight must be greater than zero.")

    # ---------------- Core Logic ----------------

    def normalized_weights(self) -> Tuple[float, float, float, float]:
        """
        Returns normalized weights (cosine, jaccard, euclidean, pearson)
        such that they sum to 1.0.
        """

        total = self.weight_cosine + self.weight_jaccard + self.weight_euclidean + self.weight_pearson

        return (
            round(self.weight_cosine / total, 6),
            round(self.weight_jaccard / total, 6),
            round(self.weight_euclidean / total, 6),
            round(self.weight_pearson / total, 6),
        )


if __name__ == "__main__":
    cfg = FSISConfig()

    print("FSIS Configuration")
    print("-------------------")
    print(f"Engine Version: {cfg.engine_version}")
    print(f"Threshold: {cfg.threshold}")
    print(f"Raw Weights: {cfg.weight_cosine}, {cfg.weight_jaccard}, {cfg.weight_euclidean}")
    print(f"Normalized Weights: {cfg.normalized_weights()}")
