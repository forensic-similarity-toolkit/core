# engine/session.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FSISRunSummary:
    run_id: str
    timestamp_utc: str
    threshold: float
    weights: Dict[str, float]
    n_samples: int
    n_edges: int


@dataclass
class FSISCaseSession:
    case_name: str
    created_utc: str = field(default_factory=_utc_now_iso)
    last_modified_utc: str = field(default_factory=_utc_now_iso)

    # Dataset traceability
    dataset_path: str = ""
    dataset_hash: str = ""

    # Current config used in GUI
    config: Dict[str, Any] = field(default_factory=dict)

    # Optional notes
    investigator_notes: str = ""

    # Run history
    runs: List[FSISRunSummary] = field(default_factory=list)

    schema_version: str = "1.0"

    def touch(self) -> None:
        self.last_modified_utc = _utc_now_iso()

    def set_dataset(self, dataset_path: str, dataset_hash: str) -> None:
        self.dataset_path = str(dataset_path)
        self.dataset_hash = str(dataset_hash)
        self.touch()

    def set_config(self, cfg: Any) -> None:
        # Accept dataclass config or plain dict
        if hasattr(cfg, "__dataclass_fields__"):
            self.config = asdict(cfg)
        elif isinstance(cfg, dict):
            self.config = dict(cfg)
        else:
            self.config = getattr(cfg, "__dict__", {}) or {}
        self.touch()

    def add_run(self, summary: FSISRunSummary) -> None:
        self.runs.append(summary)
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_name": self.case_name,
            "created_utc": self.created_utc,
            "last_modified_utc": self.last_modified_utc,
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
            "config": self.config,
            "investigator_notes": self.investigator_notes,
            "runs": [asdict(r) for r in self.runs],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FSISCaseSession":
        sess = FSISCaseSession(
            case_name=d.get("case_name", "UnnamedCase"),
            created_utc=d.get("created_utc", _utc_now_iso()),
            last_modified_utc=d.get("last_modified_utc", _utc_now_iso()),
            dataset_path=d.get("dataset_path", ""),
            dataset_hash=d.get("dataset_hash", ""),
            config=d.get("config", {}),
            investigator_notes=d.get("investigator_notes", ""),
            schema_version=d.get("schema_version", "1.0"),
        )

        runs: List[FSISRunSummary] = []
        for r in d.get("runs", []):
            runs.append(
                FSISRunSummary(
                    run_id=str(r.get("run_id", "")),
                    timestamp_utc=str(r.get("timestamp_utc", _utc_now_iso())),
                    threshold=float(r.get("threshold", 0.0)),
                    weights=dict(r.get("weights", {})),
                    n_samples=int(r.get("n_samples", 0)),
                    n_edges=int(r.get("n_edges", 0)),
                )
            )

        sess.runs = runs
        return sess


class FSISSessionStore:
    """
    Handles reading/writing session JSON files.
    """

    def __init__(self, base_dir: str = "exports"):
        self.base_dir = Path(base_dir)

    def save(self, session: FSISCaseSession, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        session.touch()

        path = folder / "session.json"
        path.write_text(
            json.dumps(session.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load(self, session_json_path: str | Path) -> FSISCaseSession:
        p = Path(session_json_path)
        d = json.loads(p.read_text(encoding="utf-8"))
        return FSISCaseSession.from_dict(d)
