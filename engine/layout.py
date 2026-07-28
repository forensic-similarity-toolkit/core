# engine/layout.py

from __future__ import annotations

import networkx as nx
from typing import Dict, Tuple


def deterministic_layout(
    G: nx.Graph,
    seed: int = 42,
    k: float | None = None,
    iterations: int = 200,
) -> Dict[str, Tuple[float, float]]:
    """
    Deterministic node positions using spring_layout with a fixed seed.
    Returns: {node: (x, y)}
    """
    pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)
    # Convert to plain floats for JSON/export stability
    return {str(n): (float(x), float(y)) for n, (x, y) in pos.items()}
