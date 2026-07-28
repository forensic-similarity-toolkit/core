# engine/cli.py

"""
FSIS CLI Runner (v1.0 Scope A)
------------------------------
Runs the engine and exports forensic artifacts to /exports/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from .config import FSISConfig
from .joint_similarity import FSISEngine
from .exporter import FSISExporter


def main():
    parser = argparse.ArgumentParser(description="FSIS v1.0 (Scope A) - Joint Similarity Engine")
    parser.add_argument("--data", required=True, help="Path to dataset CSV/TSV")
    parser.add_argument("--case", required=True, help="Case name (folder prefix)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Joint similarity threshold (0-1)")
    parser.add_argument("--wcos", type=float, default=0.33, help="Cosine weight")
    parser.add_argument("--wjac", type=float, default=0.33, help="Jaccard weight")
    parser.add_argument("--weuc", type=float, default=0.34, help="Euclidean weight")

    parser.add_argument("--export-joint", action="store_true", help="Export joint similarity matrix (CSV)")
    parser.add_argument("--export-features", action="store_true", help="Export feature matrix (CSV)")
    args = parser.parse_args()

    cfg = FSISConfig(
        weight_cosine=args.wcos,
        weight_jaccard=args.wjac,
        weight_euclidean=args.weuc,
        threshold=args.threshold,
    )

    engine = FSISEngine(cfg)
    res = engine.run(args.data)

    exporter = FSISExporter(base_dir="exports")
    exported = exporter.export(
        case_name=args.case,
        run_id=res["run_id"],
        audit=res["audit"],
        config=asdict(cfg),
        edges=res["edges"],
        joint_similarity_matrix=res["joint_similarity_matrix"],
        feature_matrix=res["feature_matrix"],
        include_joint_matrix=args.export_joint,
        include_feature_matrix=args.export_features,
    )

    print("FSIS export complete:")
    for k, v in exported.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
