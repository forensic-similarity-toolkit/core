# engine/interpreter.py

from __future__ import annotations

from typing import Any, Dict, List


# =========================================================
# NORMALIZATION
# =========================================================


def _norm(s: Any) -> str:
    return str(s).strip().lower() if s is not None else ""


# =========================================================
# ROLE CLASSIFICATION (FORENSIC CORE)
# =========================================================


def _role_bucket(role: str, category: str = "", subcategory: str = "") -> str:
    """
    Convert dictionary role/category labels into stable interpretation buckets.

    Important: the dataset category "Precursor / Impurity" is intentionally broad.
    We only classify as precursor when the role/subcategory explicitly indicates
    precursor/process chemistry. Otherwise broad precursor/impurity markers are
    treated as impurity/route markers to avoid overclaiming precursor evidence.
    """
    role_text = _norm(role)
    cat_text = _norm(category)
    sub_text = _norm(subcategory)
    text = " | ".join([role_text, cat_text, sub_text])

    # Analytical artefacts should not be promoted into active/background classes.
    if any(k in text for k in ["artefact", "artifact", "derivatization", "derivative artefact"]):
        return "artefact"

    # Non-active/background must be checked before active because "non-active" contains "active".
    if any(k in text for k in [
        "non-active", "non active", "background", "diluent", "carrier", "filler",
        "excipient", "plasticizer", "contaminant", "low-specificity", "low specificity"
    ]):
        return "diluent"

    if any(k in text for k in [
        "active pharmaceutical ingredient", "active compound", "api", "psychoactive",
        "principal drug", "drug of abuse", "primary psychoactive", "pharmacological substance"
    ]):
        return "active"

    if any(k in text for k in ["adulterant", "cutting agent", "cutting", "local anaesthetic", "local anesthetic"]):
        return "adulterant"

    # Explicit precursor/process language only.
    if any(k in role_text or k in sub_text for k in [
        "precursor", "process", "route", "synthesis route", "synthetic route", "intermediate"
    ]):
        return "precursor"

    # Broad marker/impurity/degradation labels.
    if any(k in text for k in [
        "impurity", "marker", "route marker", "source marker", "by-product", "byproduct",
        "degradant", "degradation", "metabolite", "alkaloid", "hydrolysis", "processing"
    ]):
        return "impurity"

    return "unclassified"


def _count_buckets(records: List[Any]) -> Dict[str, int]:
    counts = {
        "active": 0,
        "adulterant": 0,
        "diluent": 0,
        "precursor": 0,
        "impurity": 0,
        "artefact": 0,
        "unclassified": 0,
    }

    for rec in records:
        bucket = _role_bucket(
            getattr(rec, "role", ""),
            getattr(rec, "category", ""),
            getattr(rec, "subcategory", ""),
        )
        counts[bucket] += 1

    return counts


def _record_key(rec: Any) -> str:
    """Support both substance_categories.py records and older interpreter drafts."""
    return _norm(getattr(rec, "substance", None) or getattr(rec, "name", None) or "")


def _record_label(rec: Any) -> str:
    return str(getattr(rec, "substance", None) or getattr(rec, "name", None) or "").strip()


# =========================================================
# METRIC INTERPRETATION
# =========================================================


def _strength(joint: float) -> str:
    if joint >= 0.85:
        return "Very strong similarity"
    if joint >= 0.70:
        return "Strong similarity"
    if joint >= 0.50:
        return "Moderate similarity"
    if joint >= 0.30:
        return "Weak similarity"
    return "Very weak similarity"


def _pattern(cos: float, jac: float, euc: float) -> str:
    if cos > 0.80 and jac > 0.80 and euc > 0.80:
        return "Strong agreement across composition shape, substance overlap, and proportional closeness"
    if jac > 0.80 and euc < 0.50:
        return "High substance overlap, but large proportional differences"
    if cos > 0.80 and jac < 0.50:
        return "Similar profile shape, but different substance composition"
    if jac < 0.40 and euc < 0.40 and cos < 0.40:
        return "Low similarity across all metric components"
    return "Mixed metric behaviour"


# =========================================================
# DRIVER / SIGNIFICANCE / COVERAGE
# =========================================================


def _driver(shared_counts: Dict[str, int]) -> str:
    ranked = [
        ("Active-compound dominated", shared_counts.get("active", 0)),
        ("Adulterant/cutting-agent dominated", shared_counts.get("adulterant", 0)),
        ("Impurity/route-marker dominated", shared_counts.get("impurity", 0)),
        ("Precursor/process-marker dominated", shared_counts.get("precursor", 0)),
        ("Non-active/background dominated", shared_counts.get("diluent", 0)),
        ("Analytical artefact dominated", shared_counts.get("artefact", 0)),
        ("Unclassified-substance dominated", shared_counts.get("unclassified", 0)),
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_label, top_count = ranked[0]
    second_count = ranked[1][1]

    if top_count == 0:
        return "No clear shared driver"
    if second_count == top_count and top_count > 0:
        return "Mixed forensic driver"
    return top_label


def _significance(driver: str, strength: str, counts: Dict[str, int], coverage: str) -> str:
    active = counts.get("active", 0)
    adulterant = counts.get("adulterant", 0)
    diluent = counts.get("diluent", 0)
    impurity = counts.get("impurity", 0)
    precursor = counts.get("precursor", 0)
    artefact = counts.get("artefact", 0)

    if coverage == "Low dictionary coverage":
        return "Dictionary-limited review priority"

    if strength in {"Weak similarity", "Very weak similarity"}:
        return "Low review priority"

    if active >= 1 and strength in {"Very strong similarity", "Strong similarity"}:
        return "High review priority"

    if "Active-compound" in driver and strength == "Moderate similarity":
        return "Moderate review priority"

    if adulterant >= 1 and strength in {"Very strong similarity", "Strong similarity"}:
        return "Moderate review priority"

    if impurity >= 1 or precursor >= 1:
        return "Context-dependent review priority"

    if active == 0 and adulterant == 0 and (diluent >= 1 or artefact >= 1):
        return "Low review priority"

    return "Context-dependent review priority"


def _dictionary_coverage(counts: Dict[str, int], total: int) -> str:
    if total == 0:
        return "Low dictionary coverage"

    classified = total - counts.get("unclassified", 0)
    ratio = classified / total

    if ratio >= 0.80:
        return "High dictionary coverage"
    if ratio >= 0.50:
        return "Moderate dictionary coverage"
    return "Low dictionary coverage"


# Backward-compatible alias for older callers/tests.
def _confidence(counts: Dict[str, int], total: int) -> str:
    return _dictionary_coverage(counts, total)


def _recommendation(driver: str, strength: str, significance: str, coverage: str) -> str:
    if coverage == "Low dictionary coverage":
        return "Manual review recommended because shared substances have limited dictionary classification"

    if significance == "High review priority" and strength in {"Very strong similarity", "Strong similarity"}:
        return "Prioritise for linkage review with case context"

    if "Adulterant" in driver:
        return "Check whether shared cutting/adulterant pattern is meaningful in cluster and case context"

    if "Impurity" in driver or "Precursor" in driver:
        return "Review route/source-marker pattern against neighbouring samples and analytical context"

    if "Non-active" in driver or "artefact" in driver.lower():
        return "Interpret cautiously because similarity is dominated by non-active/background or artefact substances"

    if strength in {"Weak similarity", "Very weak similarity"}:
        return "Low operational priority unless supported by external case context"

    return "Compare with neighbouring samples, cluster membership, and case metadata"


# =========================================================
# SAMPLE SUMMARY
# =========================================================


def summarize_sample(records: List[Any]) -> Dict[str, Any]:
    counts = _count_buckets(records)
    total = len(records)
    classified = total - counts.get("unclassified", 0)

    return {
        "total_substances": total,
        "classified_substances": classified,
        "unclassified_substances": counts.get("unclassified", 0),
        "counts": counts,
    }


# =========================================================
# MAIN API
# =========================================================


def interpret_pair(
    *,
    cosine: float,
    jaccard: float,
    euclidean: float,
    joint: float,
    sample_a_records: List[Any],
    sample_b_records: List[Any],
    pearson: float = 0.0,
) -> Dict[str, Any]:
    """
    Build a cautious forensic interpretation for a sample pair by combining:
    - similarity metrics
    - shared substance forensic roles from the substance dictionary

    The interpretation is a triage/review aid. It does not assert that two
    samples are linked by itself.
    """

    map_a = {}
    for r in sample_a_records:
        key = _record_key(r)
        if key and key not in map_a:
            map_a[key] = r

    map_b = {}
    for r in sample_b_records:
        key = _record_key(r)
        if key and key not in map_b:
            map_b[key] = r

    shared_names = sorted(set(map_a.keys()) & set(map_b.keys()))
    shared_records = [map_a[name] for name in shared_names]

    counts = _count_buckets(shared_records)

    strength = _strength(float(joint))
    driver = _driver(counts)
    coverage = _dictionary_coverage(counts, len(shared_records))
    significance = _significance(driver, strength, counts, coverage)
    pattern = _pattern(float(cosine), float(jaccard), float(euclidean))
    recommendation = _recommendation(driver, strength, significance, coverage)

    interpretation = strength
    if driver == "Non-active/background dominated" and strength in {"Very strong similarity", "Strong similarity"}:
        interpretation += " (non-active/background dominated)"
    elif driver == "Adulterant/cutting-agent dominated" and strength in {"Very strong similarity", "Strong similarity"}:
        interpretation += " (adulterant/cutting-agent pattern)"
    elif driver == "Active-compound dominated" and strength in {"Very strong similarity", "Strong similarity"}:
        interpretation += " (active-compound pattern)"
    elif driver in {"Impurity/route-marker dominated", "Precursor/process-marker dominated"} and strength in {"Very strong similarity", "Strong similarity"}:
        interpretation += " (marker pattern)"

    return {
        "interpretation": interpretation,
        "driver": driver,
        "significance": significance,
        "confidence": coverage,  # retained key for backward compatibility
        "dictionary_coverage": coverage,
        "pattern": pattern,
        "recommendation": recommendation,
        "shared_total": len(shared_records),
        "shared_counts": counts,
        "shared_substances": [_record_label(r) for r in shared_records],
        "sample_a_summary": summarize_sample(sample_a_records),
        "sample_b_summary": summarize_sample(sample_b_records),
        "metrics": {
            "cosine": float(cosine),
            "jaccard": float(jaccard),
            "euclidean": float(euclidean),
            "pearson": float(pearson),
            "joint": float(joint),
        },
    }
