# precompute_similarity_matrices.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# =========================
# SETTINGS
# =========================

INPUT_FILE = r"D:\FSIS\data\modified_drug_data_set\Split files\dataset_cleaned_substance_names 2_Amphetamine.xlsx"

OUTPUT_DIR = r"D:\FSIS\data\modified_drug_data_set\Amphetamine"

SAMPLE_COL = "Sample Name"
SUBSTANCE_COL = "Substance"
VALUE_COL = "Substance percentage"
CATEGORY_COL = "Expect Substance"   # change to "Expected Substance" if needed

CATEGORY_NAME = "Amphetamine"

PRESENCE_EPS = 0.0

WEIGHT_COSINE = 1 / 3
WEIGHT_JACCARD = 1 / 3
WEIGHT_EUCLIDEAN = 1 / 3


# =========================
# HELPERS
# =========================

def normalize_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    row_sums = matrix.sum(axis=1)
    return matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)


def compute_jaccard_matrix(matrix: pd.DataFrame, presence_eps: float = 0.0) -> pd.DataFrame:
    binary = (matrix > presence_eps).astype(int).values

    intersection = binary @ binary.T
    row_sums = binary.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection

    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0
    )

    return pd.DataFrame(jaccard, index=matrix.index, columns=matrix.index)


def compute_euclidean_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    distances = euclidean_distances(matrix.values)
    similarity = 1 / (1 + distances)

    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)


def geometric_joint_similarity(cosine_df, jaccard_df, euclidean_df):
    joint = (
        np.power(cosine_df.clip(lower=0), WEIGHT_COSINE)
        * np.power(jaccard_df.clip(lower=0), WEIGHT_JACCARD)
        * np.power(euclidean_df.clip(lower=0), WEIGHT_EUCLIDEAN)
    )

    return joint


# =========================
# MAIN
# =========================

def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)

    required_cols = [SAMPLE_COL, SUBSTANCE_COL, VALUE_COL, CATEGORY_COL]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")

    # Filter only selected drug category
    df = df[
        df[CATEGORY_COL].astype(str).str.strip().str.lower()
        == CATEGORY_NAME.lower()
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for category: {CATEGORY_NAME}")

    # Clean numeric percentage values
    df[VALUE_COL] = (
        df[VALUE_COL]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.strip()
    )

    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce").fillna(0.0)

    # Build sample × substance matrix
    feature_matrix = df.pivot_table(
        index=SAMPLE_COL,
        columns=SUBSTANCE_COL,
        values=VALUE_COL,
        aggfunc="sum",
        fill_value=0.0
    )

    feature_matrix = feature_matrix.sort_index().sort_index(axis=1)

    # Normalize rows to composition proportions
    feature_matrix_norm = normalize_rows(feature_matrix)

    # Cosine similarity
    cosine = cosine_similarity(feature_matrix_norm.values)
    cosine_df = pd.DataFrame(
        cosine,
        index=feature_matrix_norm.index,
        columns=feature_matrix_norm.index
    )

    # Jaccard similarity
    jaccard_df = compute_jaccard_matrix(feature_matrix_norm, PRESENCE_EPS)

    # Euclidean similarity
    euclidean_df = compute_euclidean_similarity_matrix(feature_matrix_norm)

    # Joint similarity
    joint_df = geometric_joint_similarity(
        cosine_df,
        jaccard_df,
        euclidean_df
    )

    # Save outputs
    feature_matrix.to_csv(output_path / "feature_matrix_raw.csv")
    feature_matrix_norm.to_csv(output_path / "feature_matrix_normalized.csv")

    cosine_df.to_csv(output_path / "cosine_similarity.csv")
    jaccard_df.to_csv(output_path / "jaccard_similarity.csv")
    euclidean_df.to_csv(output_path / "euclidean_similarity.csv")
    joint_df.to_csv(output_path / "joint_similarity_equal_weights.csv")

    print("Precomputation complete.")
    print(f"Category: {CATEGORY_NAME}")
    print(f"Samples: {feature_matrix.shape[0]}")
    print(f"Substances: {feature_matrix.shape[1]}")
    print(f"Output folder: {output_path}")


if __name__ == "__main__":
    main()
