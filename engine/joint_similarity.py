# engine/joint_similarity.py

"""
FSIS Engine Entry Point (Scope A v1.0)
--------------------------------------
One-call execution:
load dataset -> compute real-time metrics -> joint similarity -> threshold network -> audit.
"""

from __future__ import annotations

from typing import Dict, Any

from .config import FSISConfig
from .data_loader import FSISDataLoader
from .metrics import FSISMetrics
from .network_builder import FSISNetworkBuilder
from .audit import FSISAudit


class FSISEngine:
    def __init__(self, config: FSISConfig):
        self.config = config

    def run(self, dataset_path: str) -> Dict[str, Any]:
        # 1) Load + validate
        loader = FSISDataLoader(dataset_path)
        X, df_raw, loader_meta = loader.load()

        # 2) Compute joint similarity immediately (real-time)
        metrics = FSISMetrics(X)
        w = self.config.normalized_weights()
        joint_df = metrics.joint_similarity(weights=w)

        # 3) Audit (Run ID, hashing, parsing metadata)
        audit = FSISAudit(
            feature_matrix=X,
            weights=w,
            threshold=self.config.threshold,
            loader_meta=loader_meta,
        )
        audit_record = audit.get_audit_record()

        # 4) Thresholded network
        builder = FSISNetworkBuilder(joint_similarity_df=joint_df, threshold=self.config.threshold)
        graph_meta = {
            "run_id": audit_record["run_id"],
            "engine_version": audit_record["engine_version"],
            "engine_name": audit_record["engine_name"],
        }
        G = builder.build_graph(graph_meta=graph_meta)
        edges_df = builder.get_edge_list()

        # 5) Package results
        return {
            "run_id": audit_record["run_id"],
            "audit": audit_record,
            "feature_matrix": X,
            "joint_similarity_matrix": joint_df,
            "edges": edges_df,
            "graph": G,
            "raw_dataset": df_raw,          # kept for future modules / reporting
            "loader_meta": loader_meta,
        }
