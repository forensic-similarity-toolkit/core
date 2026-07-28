# engine/network_builder.py

"""
FSIS Network Builder (Forensic-grade)
------------------------------------
Builds a NetworkX graph from a joint similarity matrix using threshold filtering.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from typing import Dict, Any, Optional


class FSISNetworkBuilder:
    def __init__(self, joint_similarity_df: pd.DataFrame, threshold: float):
        if joint_similarity_df is None or joint_similarity_df.empty:
            raise ValueError("Joint similarity matrix is empty.")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")

        self.js_df = joint_similarity_df.copy()
        self.samples = self.js_df.index.astype(str).tolist()
        self.threshold = round(float(threshold), 6)
        self.graph: Optional[nx.Graph] = None

    def build_graph(self, graph_meta: Optional[Dict[str, Any]] = None) -> nx.Graph:
        """
        Create an undirected graph where edges represent sample pairs with
        joint similarity >= threshold.
        """
        G = nx.Graph()
        G.add_nodes_from(self.samples)

        # Attach forensic metadata to graph
        G.graph["fsis"] = graph_meta or {}
        G.graph["fsis"]["threshold"] = self.threshold

        # Upper triangle only for deterministic edges
        for i, a in enumerate(self.samples):
            for j in range(i + 1, len(self.samples)):
                b = self.samples[j]
                sim = float(self.js_df.iloc[i, j])
                sim_rounded = round(sim, 6)

                # FSIS rule: zero-similarity pairs must never become edges.
                # Even when the threshold is set to 0, samples with JS = 0
                # remain in the graph as isolated nodes.
                if sim_rounded > 0.0 and sim_rounded >= self.threshold:
                    G.add_edge(a, b, joint_similarity=sim_rounded)

        self.graph = G
        return G

    def get_edge_list(self) -> pd.DataFrame:
        """
        Returns a deterministic edge list:
        Sample_A, Sample_B, joint_similarity
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")

        rows = []
        for u, v, d in self.graph.edges(data=True):
            a, b = (u, v) if u <= v else (v, u)
            rows.append((a, b, float(d["joint_similarity"])))

        df_edges = pd.DataFrame(rows, columns=["Sample_A", "Sample_B", "joint_similarity"])
        df_edges = df_edges.sort_values(["Sample_A", "Sample_B"]).reset_index(drop=True)
        df_edges["joint_similarity"] = df_edges["joint_similarity"].round(6)
        return df_edges
