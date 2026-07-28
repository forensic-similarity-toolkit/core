# FSIS — Forensic Similarity Intelligence System

FSIS is an open-source, browser-based expert system for comparing the chemical
composition profiles of drug samples and surfacing candidate common-source or
common-supply relationships as an interactive similarity network.

It combines five complementary pairwise similarity metrics — cosine, Jaccard,
Euclidean, TF-IDF cosine, and Pearson correlation — into a single configurable
joint similarity score via a weighted geometric product, and exposes the
resulting network through a web interface with no programming required:
analysis presets, threshold sweeps, per-cluster chemical fingerprinting,
temporal analysis, and centrality/community-detection dashboards.

FSIS's design, validation against published European drug market
intelligence, and benchmark against standard machine learning clustering
methods are described in the accompanying paper (citation to be added once
published).

## Running it

```bash
pip install -r requirements.txt
uvicorn web.main:app --reload
```

Then open `http://localhost:8000` in a browser and select **Synthetic_Demo**
from the dataset dropdown.

## Data

This repository ships a **synthetic** demonstration dataset
(`data/Synthetic_Demo/`) — fabricated sample IDs and proportions across four
drug categories (heroin, cocaine, MDMA, ketamine), built from generic public
chemical-substance names. It is provided purely so the tool can be tried out
immediately; it does not represent real forensic samples.

The reference dataset used in the accompanying paper (317 real samples
collected by Kosmicare Portugal, a drug-checking and harm-reduction
organisation) is not included in this repository. It is available on request
from the corresponding author; anyone using it is asked to cite the paper and
acknowledge Kosmicare Portugal as the source of the underlying drug samples.

## Loading your own data

Each dataset is a folder under `data/` containing:

- `feature_matrix_wide_normalized.csv` — sample × substance matrix, row-normalised proportions, first column = sample ID
- `cosine_similarity.csv`, `jaccard_similarity.csv`, `euclidean_similarity.csv` — precomputed pairwise similarity matrices
- `sample_metadata.csv` — first column = sample ID, plus any metadata columns (a `Year`/`Date` column enables the temporal analysis dashboard)

`preprocess-code/precompute_similarity_matrices.py` builds these files from a
raw sample × substance spreadsheet — edit the `SETTINGS` block at the top of
the script and run it directly.

## Repository layout

- `engine/` — computation core: data loading, similarity metrics, network construction, session/audit/export utilities
- `web/` — the FastAPI + Plotly web application described above
- `preprocess-code/` — script for building a precomputed dataset folder from raw source data
- `data/substance_categories.csv` — maps substance names to forensic roles (true substance / substitute / adulterant / diluent / impurity / contaminant)

## License

MIT — see `LICENSE`.
