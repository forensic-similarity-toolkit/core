# preprocess-code/process_bb_dataset.py
#
# Converts DATA_BB-New_categorisations.xlsx (wide two-header-row format)
# into per-category precomputed similarity matrices and a substance dictionary.
#
# Outputs written to data/
#   data/{Category}/   - similarity matrices, feature matrices, metadata
#   data/Dataset/      - all samples combined
#   data/substance_categories.csv

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from engine.preprocessor import FSISPreprocessor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = ROOT / "data" / "DATA_BB-New_categorisations.xlsx"
OUTPUT_BASE = ROOT / "data"

# Forensic role column pairs (name_col_index, pct_col_index)
# Derived from the two-row header structure of the source file.
ROLE_COLUMN_PAIRS: list[tuple[str, list[tuple[int, int]]]] = [
    ("True Substance", [(6, 7), (8, 9), (10, 11), (12, 13)]),
    ("Substitute",     [(14, 15), (16, 17), (18, 19), (20, 21)]),
    ("Adulterant",     [(22, 23), (24, 25), (26, 27), (28, 29)]),
    ("Diluent",        [(30, 31)]),
    ("Impurity",       [(32, 33), (34, 35), (36, 37), (38, 39),
                        (40, 41), (42, 43), (44, 45), (46, 47),
                        (48, 49), (50, 51), (52, 53)]),
    ("Contaminants",   [(54, 55), (56, 57)]),
    # Cols 58 (Fatty acids) and 59 (Unknown) are boolean flags — skipped.
]

# Meta columns (column index → output name)
META_COLS: dict[int, str] = {
    1: "Sample Name",
    2: "Type of sample",
    3: "Technique",
    4: "Date/year of reception",
    5: "Expected Substance",
}

# Normalise Expected Substance values → folder name
CATEGORY_MAP: dict[str, str] = {
    "alucinogénios": "Hallucinogens",
    "alucinogenios": "Hallucinogens",
    "benzodizepina": "Benzodiazepines",
    "cocaine":       "Cocaine",
    "nootropic":     "Nootropic",
    "pink cocaine":  "Pink_cocaine",
    "pink star":     "Pink_Star",
    "speedball":     "Speedball",
    "2c-b":          "2C-B",
    "amphetamine":   "Amphetamine",
    "benzodiazepines": "Benzodiazepines",
    "cannabinoids":  "Cannabinoids",
    "dmt":           "DMT",
    "heroin":        "Heroin",
    "ketamine":      "Ketamine",
    "lsd":           "LSD",
    "mdma":          "MDMA",
}

# Forensic role → substance_categories.csv Category label
ROLE_TO_CATEGORY: dict[str, str] = {
    "True Substance": "Active Pharmaceutical Ingredient",
    "Substitute":     "Substitute",
    "Adulterant":     "Adulterant",
    "Diluent":        "Diluent",
    "Impurity":       "Precursor / Impurity",
    "Contaminants":   "Contaminants",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean_str(v) -> str:
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() in ("nan", "none", "false", "true") else s


def _clean_pct(v) -> float:
    s = str(v).replace("%", "").replace(">", "").replace(",", "").strip()
    try:
        return max(0.0, float(s))
    except (ValueError, TypeError):
        return 0.0


def normalise_category(raw: str) -> str | None:
    """Return the canonical folder name, or None to skip."""
    key = raw.strip().lower()
    return CATEGORY_MAP.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# PARSE WIDE FORMAT → LONG FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def parse_wide_to_long(filepath: Path) -> pd.DataFrame:
    """
    Read the two-header-row Excel file and return a tidy long-format DataFrame:
        Sample Name | Type of sample | Technique | Date/year of reception
        | Expected Substance | Substance | AREA % | Forensic_Role
    Only rows where Substance is a non-empty, non-false string are kept.
    Expected Substance is normalised; rows with unrecognised categories are dropped.
    """
    df_raw = pd.read_excel(filepath, header=None)
    data = df_raw.iloc[2:].reset_index(drop=True)   # skip the two header rows

    rows: list[dict] = []

    for _, row in data.iterrows():
        # Meta
        sample   = _clean_str(row.get(1, ""))
        type_smp = _clean_str(row.get(2, ""))
        tech     = _clean_str(row.get(3, ""))
        date     = _clean_str(row.get(4, ""))
        raw_cat  = _clean_str(row.get(5, ""))

        if not sample:
            continue

        category = normalise_category(raw_cat)
        if category is None:
            continue   # Unknown or unrecognised

        meta = {
            "Sample Name": sample,
            "Type of sample": type_smp,
            "Technique": tech,
            "Date/year of reception": date,
            "Expected Substance": category,
        }

        # Substance pairs
        for role, pairs in ROLE_COLUMN_PAIRS:
            for name_col, pct_col in pairs:
                substance = _clean_str(row.get(name_col, ""))
                if not substance:
                    continue
                pct = _clean_pct(row.get(pct_col, 0))
                rows.append({**meta, "Substance": substance, "AREA %": pct, "Forensic_Role": role})

    long_df = pd.DataFrame(rows)
    print(f"  Parsed {len(long_df)} substance observations across "
          f"{long_df['Sample Name'].nunique()} samples in "
          f"{long_df['Expected Substance'].nunique()} categories.")
    return long_df


# ─────────────────────────────────────────────────────────────────────────────
# SUBSTANCE DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def build_substance_categories(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique substance, determine the most-common forensic role
    across all samples and return a substance_categories.csv DataFrame.
    """
    # Count occurrences of each (substance, role) pair
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in long_df.iterrows():
        sub  = str(row["Substance"]).strip()
        role = str(row["Forensic_Role"]).strip()
        if sub and role:
            counts[sub][role] += 1

    records = []
    for substance, role_counts in sorted(counts.items(), key=lambda x: x[0].lower()):
        top_role = max(role_counts, key=role_counts.get)
        category = ROLE_TO_CATEGORY.get(top_role, "Unknown/Unclassified")
        records.append({
            "Substance": substance,
            "Category": category,
            "Subcategory": top_role,
            "Role": top_role,
            "Notes": "",
        })

    return pd.DataFrame(records, columns=["Substance", "Category", "Subcategory", "Role", "Notes"])


# ─────────────────────────────────────────────────────────────────────────────
# PER-CATEGORY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_all(long_df: pd.DataFrame, output_base: Path, source_file: str) -> None:
    pre = FSISPreprocessor()
    categories = sorted(long_df["Expected Substance"].unique())

    results: list[dict] = []

    for cat in categories:
        cat_df = long_df[long_df["Expected Substance"] == cat].copy()
        n_samples = cat_df["Sample Name"].nunique()
        print(f"\n{'─'*60}")
        print(f"Category: {cat}  ({n_samples} samples)")

        if n_samples < 2:
            print(f"  SKIP: fewer than 2 samples (need >= 2 for similarity matrices).")
            continue

        out_dir = output_base / cat
        try:
            meta = pre.process_category(
                df=cat_df,
                category_name=cat,
                sample_col="Sample Name",
                substance_col="Substance",
                value_col="AREA %",
                category_col="Expected Substance",
                output_dir=out_dir,
                source_file=source_file,
                progress_cb=lambda m: print(f"  {m}"),
            )
            results.append(meta)
            print(f"  ✅ Done: {meta['n_samples']} samples, {meta['n_substances']} substances")
        except Exception as exc:
            print(f"  ❌ Failed: {exc}")

    # ── Dataset (all samples combined) ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Category: Dataset  (all {long_df['Sample Name'].nunique()} samples combined)")
    out_dir_ds = output_base / "Dataset"
    try:
        meta_ds = pre.process_category(
            df=long_df,
            category_name="Dataset",
            sample_col="Sample Name",
            substance_col="Substance",
            value_col="AREA %",
            category_col=None,          # no category filter — use all rows
            output_dir=out_dir_ds,
            source_file=source_file,
            progress_cb=lambda m: print(f"  {m}"),
        )
        print(f"  ✅ Done: {meta_ds['n_samples']} samples, {meta_ds['n_substances']} substances")
    except Exception as exc:
        print(f"  ❌ Dataset failed: {exc}")

    print(f"\n{'═'*60}")
    print(f"Processed {len(results)} categories successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_BASE}")
    print()

    print("Parsing wide-format Excel file...")
    long_df = parse_wide_to_long(INPUT_FILE)

    print("\nBuilding substance dictionary...")
    substance_df = build_substance_categories(long_df)
    out_path = OUTPUT_BASE / "substance_categories.csv"
    substance_df.to_csv(out_path, index=False)
    print(f"  ✅ Saved {len(substance_df)} substances → {out_path}")

    print("\nProcessing per-category similarity matrices...")
    process_all(long_df, OUTPUT_BASE, source_file=str(INPUT_FILE))

    print("\nDone.")


if __name__ == "__main__":
    main()
