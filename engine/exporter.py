# engine/exporter.py

"""
FSIS Exporter (Forensic-grade)
------------------------------
Writes deterministic outputs to a case folder:
- audit.json
- config.json
- edges.csv
Optional:
- joint_similarity.csv
- feature_matrix.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class FSISExporter:
    def __init__(self, base_dir: str = "exports"):
        self.base_dir = Path(base_dir)

    def _safe_case_name(self, case_name: str) -> str:
        # conservative filesystem-safe name
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in case_name.strip())

    def create_case_folder(self, case_name: str, run_id: str) -> Path:
        case = self._safe_case_name(case_name)
        folder = self.base_dir / f"{case}_{run_id[:12]}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def export(
        self,
        case_name: str,
        run_id: str,
        audit: Dict[str, Any],
        config: Dict[str, Any],
        edges: pd.DataFrame,
        joint_similarity_matrix: Optional[pd.DataFrame] = None,
        feature_matrix: Optional[pd.DataFrame] = None,
        include_joint_matrix: bool = False,
        include_feature_matrix: bool = False,
    ) -> Dict[str, str]:
        """
        Returns a dict of exported file paths (as strings).
        """
        folder = self.create_case_folder(case_name, run_id)

        # audit.json
        audit_path = folder / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

        # config.json
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

        # edges.csv (always)
        edges_path = folder / "edges.csv"
        edges.to_csv(edges_path, index=False)

        exported = {
            "folder": str(folder),
            "audit.json": str(audit_path),
            "config.json": str(config_path),
            "edges.csv": str(edges_path),
        }

        # Optional exports
        if include_joint_matrix:
            if joint_similarity_matrix is None:
                raise ValueError("include_joint_matrix=True but joint_similarity_matrix is None")
            js_path = folder / "joint_similarity.csv"
            joint_similarity_matrix.to_csv(js_path, index=True)
            exported["joint_similarity.csv"] = str(js_path)

        if include_feature_matrix:
            if feature_matrix is None:
                raise ValueError("include_feature_matrix=True but feature_matrix is None")
            fm_path = folder / "feature_matrix.csv"
            feature_matrix.to_csv(fm_path, index=True)
            exported["feature_matrix.csv"] = str(fm_path)

        return exported
