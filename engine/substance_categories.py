# engine/substance_categories.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import re

import pandas as pd


DEFAULT_CATEGORY_CSV = Path("data/substance_categories.csv")


@dataclass(frozen=True)
class SubstanceRecord:
    """
    Canonical substance metadata record used across FSIS.
    """
    substance: str
    category: str = "Unclassified"
    subcategory: str = ""
    role: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Substance": self.substance,
            "Category": self.category,
            "Subcategory": self.subcategory,
            "Role": self.role,
            "Notes": self.notes,
        }


class SubstanceCategoryStore:
    """
    Loads and serves forensic substance category metadata from CSV.

    Required columns:
    - Substance
    - Category

    Optional columns:
    - Subcategory
    - Role
    - Notes

    Main goals:
    - Stable lookup by substance name
    - Graceful fallback for unclassified substances
    - Compatibility with both old summary code and new interpreter logic
    """

    def __init__(self, csv_path: str | Path = DEFAULT_CATEGORY_CSV):
        self.csv_path = Path(csv_path)
        self.df = self._load_csv()
        self._records = self._build_record_map()

    # -----------------------------
    # Internal helpers
    # -----------------------------

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _norm_key(value: Any) -> str:
        text = SubstanceCategoryStore._clean_text(value).lower()
        text = text.replace("δ", "delta").replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _alias_keys(value: Any) -> set[str]:
        raw = SubstanceCategoryStore._norm_key(value)
        if not raw:
            return set()
        keys = {raw}
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if compact:
            keys.add(compact)
        for part in re.split(r"\s*(?:;|/|\bor\b|,)\s*", raw):
            part = part.strip()
            if part:
                keys.add(part)
                keys.add(re.sub(r"[^a-z0-9]+", "", part))
        manual = {
            "acetaminophen": "paracetamol",
            "acetaminophentms": "acetaminophen tms",
            "6monoacetylmorphine": "6-mam",
            "6-monoacetylmorphine": "6-mam",
            "delta9thc": "delta-9-thc",
            "delta-9-thc": "delta-8-thc; delta-9-thc",
            "tetramisoleorlevamisole": "tetramisole; levamisole",
            "tetramisole or levamisole": "tetramisole; levamisole",
            "unknownsubstances": "unknown",
            "unknown substances": "unknown",
            "nacetylnorcocaine": "n-acetylnorcocaine",
            "p phenetidine": "p-phenetidine",
        }
        for k in list(keys):
            if k in manual:
                target = manual[k]
                keys.add(target)
                keys.add(re.sub(r"[^a-z0-9]+", "", target))
        return {k for k in keys if k}

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Substance category CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required = {"Substance", "Category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in substance category CSV: {sorted(missing)}"
            )

        # Ensure optional columns exist
        for col in ("Subcategory", "Role", "Notes"):
            if col not in df.columns:
                df[col] = ""

        # Normalize text fields
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Drop blank substance rows
        df = df[df["Substance"] != ""].copy()

        # If Category is blank, treat as Unclassified instead of dropping it
        df["Category"] = df["Category"].replace("", "Unclassified")

        # Check duplicates using normalized key
        norm_substances = df["Substance"].astype(str).str.strip().str.lower()
        if norm_substances.duplicated().any():
            dups = df.loc[norm_substances.duplicated(), "Substance"].tolist()
            raise ValueError(
                f"Duplicate Substance values in category CSV (case-insensitive): {dups[:20]}"
            )

        # Stable ordering
        df = df.sort_values("Substance").reset_index(drop=True)
        return df

    def _build_record_map(self) -> Dict[str, SubstanceRecord]:
        records: Dict[str, SubstanceRecord] = {}

        for _, row in self.df.iterrows():
            substance = self._clean_text(row.get("Substance", ""))
            if not substance:
                continue

            rec = SubstanceRecord(
                substance=substance,
                category=self._clean_text(row.get("Category", "Unclassified")) or "Unclassified",
                subcategory=self._clean_text(row.get("Subcategory", "")),
                role=self._clean_text(row.get("Role", "")),
                notes=self._clean_text(row.get("Notes", "")),
            )

            for key in self._alias_keys(substance):
                records.setdefault(key, rec)

        return records

    def _fallback_record(self, substance: str) -> SubstanceRecord:
        s = self._clean_text(substance)
        return SubstanceRecord(
            substance=s,
            category="Unclassified",
            subcategory="",
            role="",
            notes="No dictionary match found.",
        )

    # -----------------------------
    # Public lookup API
    # -----------------------------

    def has_substance(self, substance: str) -> bool:
        return any(key in self._records for key in self._alias_keys(substance))

    def get_record_obj(self, substance: str) -> SubstanceRecord:
        for key in self._alias_keys(substance):
            if key in self._records:
                return self._records[key]
        return self._fallback_record(substance)

    def get_record(self, substance: str) -> Dict[str, Any]:
        """
        Backward-compatible dict-style record.
        """
        return self.get_record_obj(substance).to_dict()

    def get_category(self, substance: str) -> str:
        return self.get_record_obj(substance).category

    def get_role(self, substance: str) -> str:
        return self.get_record_obj(substance).role

    def get_subcategory(self, substance: str) -> str:
        return self.get_record_obj(substance).subcategory

    def categorize_list(self, substances: List[str]) -> Dict[str, str]:
        return {
            str(s).strip(): self.get_category(str(s))
            for s in substances
            if str(s).strip()
        }

    def get_records_for_substances(self, substances: List[str]) -> List[SubstanceRecord]:
        """
        Returns canonical SubstanceRecord objects for the given substances.
        Preserves input order after cleaning, excluding blanks.
        """
        out: List[SubstanceRecord] = []
        for s in substances:
            s_clean = self._clean_text(s)
            if not s_clean:
                continue
            out.append(self.get_record_obj(s_clean))
        return out

    # -----------------------------
    # Grouping / summary helpers
    # -----------------------------

    def group_by_category(self, substances: List[str]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for substance in substances:
            s = self._clean_text(substance)
            if not s:
                continue

            cat = self.get_category(s)
            grouped.setdefault(cat, []).append(s)

        for cat in grouped:
            grouped[cat] = sorted(grouped[cat], key=str.lower)

        return dict(sorted(grouped.items(), key=lambda x: x[0].lower()))

    def group_by_role_bucket(self, substances: List[str]) -> Dict[str, List[str]]:
        """
        Stronger forensic grouping for interpretation.
        Maps free-text Role/Category/Subcategory into stable buckets.
        """
        grouped: Dict[str, List[str]] = {
            "active": [],
            "adulterant": [],
            "diluent": [],
            "precursor": [],
            "impurity": [],
            "artefact": [],
            "unclassified": [],
        }

        for substance in substances:
            s = self._clean_text(substance)
            if not s:
                continue

            rec = self.get_record_obj(s)
            bucket = self.role_bucket_for_record(rec)
            grouped.setdefault(bucket, []).append(s)

        for k in grouped:
            grouped[k] = sorted(grouped[k], key=str.lower)

        return grouped

    def summarize_for_substances(self, substances: List[str]) -> Dict[str, Any]:
        """
        Backward-compatible category summary, with extra role-bucket summary added.
        """
        clean = sorted({self._clean_text(s) for s in substances if self._clean_text(s)}, key=str.lower)
        grouped = self.group_by_category(clean)
        role_grouped = self.group_by_role_bucket(clean)

        return {
            "n_substances": len(clean),
            "categories": grouped,
            "category_counts": {k: len(v) for k, v in grouped.items()},
            "role_buckets": role_grouped,
            "role_bucket_counts": {k: len(v) for k, v in role_grouped.items()},
            "unclassified": grouped.get("Unclassified", []),
        }

    # -----------------------------
    # Interpreter support
    # -----------------------------

    @classmethod
    def role_bucket_for_record(cls, rec: SubstanceRecord) -> str:
        """
        Convert raw dictionary fields into stable forensic buckets.
        """
        text = " | ".join(
            [
                cls._norm_key(rec.role),
                cls._norm_key(rec.category),
                cls._norm_key(rec.subcategory),
            ]
        )

        if any(k in text for k in ["active", "api", "psychoactive", "principal drug", "drug of abuse"]):
            return "active"
        if any(k in text for k in ["adulterant", "cutting agent"]):
            return "adulterant"
        if any(k in text for k in ["diluent", "carrier", "non-active", "non active", "filler"]):
            return "diluent"
        if any(k in text for k in ["precursor"]):
            return "precursor"
        if any(k in text for k in ["impurity", "by-product", "byproduct", "degradant", "degradation"]):
            return "impurity"
        if any(k in text for k in ["artefact", "artifact"]):
            return "artefact"

        return "unclassified"

    def count_role_buckets(self, substances: List[str]) -> Dict[str, int]:
        grouped = self.group_by_role_bucket(substances)
        return {k: len(v) for k, v in grouped.items()}

    def compare_substance_sets(self, substances_a: List[str], substances_b: List[str]) -> Dict[str, Any]:
        """
        Utility for pair-level interpretation support.
        """
        clean_a = {self._clean_text(s) for s in substances_a if self._clean_text(s)}
        clean_b = {self._clean_text(s) for s in substances_b if self._clean_text(s)}

        shared = sorted(clean_a & clean_b, key=str.lower)
        only_a = sorted(clean_a - clean_b, key=str.lower)
        only_b = sorted(clean_b - clean_a, key=str.lower)

        shared_counts = self.count_role_buckets(shared)

        return {
            "shared": shared,
            "only_a": only_a,
            "only_b": only_b,
            "shared_count": len(shared),
            "shared_role_bucket_counts": shared_counts,
        }
