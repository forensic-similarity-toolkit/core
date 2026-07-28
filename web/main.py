# web/main.py

"""
FSIS Web Application
--------------------
FastAPI server for the Forensic Similarity Intelligence System.
Single-user deployment on Render.
"""

from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Ensure engine is importable from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.tfidf_similarity import TFIDFSimilarity
from engine.network_builder import FSISNetworkBuilder

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(title="FSIS")

BASE_DIR = Path(__file__).parent
DATA_DIR = ROOT / "data"

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ─────────────────────────────────────────────
# In-memory state (single user)
# ─────────────────────────────────────────────

class AppState:
    dataset: str | None = None
    feature_matrix: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    cosine_sim: pd.DataFrame | None = None
    jaccard_sim: pd.DataFrame | None = None
    euclidean_sim: pd.DataFrame | None = None
    tfidf_sim: pd.DataFrame | None = None
    pearson_sim: pd.DataFrame | None = None
    joint_sim: pd.DataFrame | None = None
    # Per-compute filtered individual metrics (active sample set)
    active_cosine: pd.DataFrame | None = None
    active_jaccard: pd.DataFrame | None = None
    active_euclidean: pd.DataFrame | None = None
    active_tfidf: pd.DataFrame | None = None
    active_pearson: pd.DataFrame | None = None
    sqrt_pretreatment: bool = False
    graph: nx.Graph | None = None
    node_positions: Dict[str, List[float]] | None = None
    node_component: Dict[str, int] | None = None
    sample_years: Dict[str, int] = {}
    weights: Dict[str, float] = {
        "weight_cosine": 0.25,
        "weight_jaccard": 0.25,
        "weight_euclidean": 0.25,
        "weight_tfidf": 0.25,
        "weight_pearson": 0.0,
        "threshold": 0.5,
    }

state = AppState()

# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class LoadRequest(BaseModel):
    dataset: str

class ComputeRequest(BaseModel):
    weight_cosine: float = 0.25
    weight_jaccard: float = 0.25
    weight_euclidean: float = 0.25
    weight_tfidf: float = 0.25
    weight_pearson: float = 0.0
    threshold: float = 0.5
    sqrt_pretreatment: bool = False
    filter_chemicals: List[str] = []
    filter_mode: str = "any"   # "any" = OR logic, "all" = AND logic
    year_min: Optional[int] = None
    year_max: Optional[int] = None

class SelectionRequest(BaseModel):
    samples: List[str]

# ─────────────────────────────────────────────
# Helper: list datasets
# ─────────────────────────────────────────────

def get_datasets() -> List[str]:
    datasets = []
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir() and (item / "feature_matrix_wide_normalized.csv").exists():
                datasets.append(item.name)
    return sorted(datasets)

# ─────────────────────────────────────────────
# Helper: sqrt pre-treatment
# ─────────────────────────────────────────────

def apply_sqrt_pretreatment(fm: pd.DataFrame) -> pd.DataFrame:
    """Apply sqrt to feature values then re-normalise rows to proportions."""
    arr = np.sqrt(fm.to_numpy(dtype=np.float64))
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    arr = arr / row_sums
    return pd.DataFrame(arr, index=fm.index, columns=fm.columns)

# ─────────────────────────────────────────────
# Helpers: live metric computation
# (used when sqrt pre-treatment is active)
# ─────────────────────────────────────────────

def compute_cosine_sim(fm: pd.DataFrame) -> pd.DataFrame:
    X = fm.to_numpy(dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-12
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(np.round(sim, 6), index=fm.index, columns=fm.index)

def compute_jaccard_sim(fm: pd.DataFrame) -> pd.DataFrame:
    Xb = (fm.to_numpy(dtype=np.float64) > 0.0).astype(np.int32)
    inter = Xb @ Xb.T
    sums = Xb.sum(axis=1).reshape(-1, 1)
    union = sums + sums.T - inter
    union = np.where(union == 0, 1, union)
    sim = inter / union
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(np.round(sim, 6), index=fm.index, columns=fm.index)

def compute_euclidean_sim(fm: pd.DataFrame) -> pd.DataFrame:
    X = fm.to_numpy(dtype=np.float64)
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    sim = 1.0 / (1.0 + np.sqrt(d2))
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(np.round(sim, 6), index=fm.index, columns=fm.index)

# ─────────────────────────────────────────────
# Helper: Pearson similarity
# ─────────────────────────────────────────────

def compute_pearson_sim(fm: pd.DataFrame) -> pd.DataFrame:
    X = fm.to_numpy(dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-12
    Xn = Xc / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(np.round(sim, 6), index=fm.index, columns=fm.index)

# ─────────────────────────────────────────────
# Helper: joint similarity (5 metrics)
# ─────────────────────────────────────────────

def compute_joint(
    cos: pd.DataFrame,
    jac: pd.DataFrame,
    euc: pd.DataFrame,
    tfidf: pd.DataFrame,
    pea: pd.DataFrame,
    w_cos: float,
    w_jac: float,
    w_euc: float,
    w_tfidf: float,
    w_pea: float,
) -> pd.DataFrame:
    total = w_cos + w_jac + w_euc + w_tfidf + w_pea
    if total == 0:
        total = 1.0
    w_cos /= total
    w_jac /= total
    w_euc /= total
    w_tfidf /= total
    w_pea /= total

    def pc(mat: np.ndarray, w: float) -> np.ndarray:
        if w == 0.0:
            return np.ones_like(mat)
        return np.power(np.clip(mat, 0.0, 1.0), w)

    joint = (
        pc(cos.to_numpy(dtype=np.float64), w_cos)
        * pc(jac.to_numpy(dtype=np.float64), w_jac)
        * pc(euc.to_numpy(dtype=np.float64), w_euc)
        * pc(tfidf.to_numpy(dtype=np.float64), w_tfidf)
        * pc(pea.to_numpy(dtype=np.float64), w_pea)
    )
    joint = np.clip(joint, 0.0, 1.0)
    np.fill_diagonal(joint, 1.0)
    return pd.DataFrame(np.round(joint, 6), index=cos.index, columns=cos.columns)

# ─────────────────────────────────────────────
# Helper: build Plotly figure JSON
# ─────────────────────────────────────────────

CLUSTER_COLORS = [
    "#4a7fd4", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a",
    "#ff5722", "#607d8b", "#795548", "#ff9800", "#3f51b5",
]


def build_plotly_figure(G: nx.Graph, joint_sim: pd.DataFrame) -> Dict[str, Any]:
    if len(G.nodes()) == 0:
        return {
            "data": [],
            "layout": {"paper_bgcolor": "#f8f9fa", "plot_bgcolor": "#f8f9fa"},
        }

    samples = list(G.nodes())

    # Spring layout — deterministic
    pos = nx.spring_layout(G, seed=42, k=1.8 / max(1, np.sqrt(len(samples))))
    state.node_positions = {n: list(pos[n]) for n in samples}

    # Connected components → colour
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_map: Dict[str, int] = {}
    for i, comp in enumerate(components):
        for node in comp:
            comp_map[node] = i
    state.node_component = comp_map

    # ── Edge trace ──────────────────────────────
    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace: Dict[str, Any] = {
        "type": "scatter",
        "x": edge_x,
        "y": edge_y,
        "mode": "lines",
        "line": {"width": 1, "color": "rgba(120,140,160,0.35)"},
        "hoverinfo": "none",
        "showlegend": False,
    }

    # ── Node trace ──────────────────────────────
    node_x = [float(pos[s][0]) for s in samples]
    node_y = [float(pos[s][1]) for s in samples]
    degrees = [G.degree(s) for s in samples]
    max_deg = max(degrees) if degrees else 0
    node_sizes = [int(8 + 16 * (d / max_deg) ** 0.5) if max_deg > 0 else 10 for d in degrees]
    node_colors = [
        CLUSTER_COLORS[comp_map.get(s, 0) % len(CLUSTER_COLORS)] for s in samples
    ]

    hover_texts = []
    for s in samples:
        deg = G.degree(s)
        comp_idx = comp_map.get(s, 0)
        comp_size = len(components[comp_idx]) if comp_idx < len(components) else 1
        hover_texts.append(
            f"<b>{s}</b><br>Connections: {deg}<br>Cluster size: {comp_size}"
        )

    node_trace: Dict[str, Any] = {
        "type": "scatter",
        "x": node_x,
        "y": node_y,
        "mode": "markers+text",
        "text": samples,
        "textposition": "top center",
        "textfont": {"size": 8, "color": "#444"},
        "customdata": samples,
        "hovertext": hover_texts,
        "hoverinfo": "text",
        "marker": {
            "size": node_sizes,
            "color": node_colors,
            "line": {"width": 1.5, "color": "white"},
            "opacity": 0.9,
        },
        "showlegend": False,
    }

    layout: Dict[str, Any] = {
        "showlegend": False,
        "hovermode": "closest",
        "margin": {"b": 5, "l": 5, "r": 5, "t": 5},
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "dragmode": "pan",
    }

    return {"data": [edge_trace, node_trace], "layout": layout}

# ─────────────────────────────────────────────
# Helper: graph stats
# ─────────────────────────────────────────────

def graph_stats(G: nx.Graph) -> Dict[str, int]:
    components = list(nx.connected_components(G))
    isolates = sum(1 for c in components if len(c) == 1)
    return {
        "n_samples": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "n_isolates": isolates,
        "n_clusters": len([c for c in components if len(c) > 1]),
    }

# ─────────────────────────────────────────────
# Helper: sample bar chart
# ─────────────────────────────────────────────

def build_sample_chart(sample_name: str, feature_matrix: pd.DataFrame) -> Dict[str, Any]:
    row = feature_matrix.loc[sample_name]
    present = row[row > 0.0].sort_values(ascending=True)  # ascending for horizontal bar

    if present.empty:
        return {"data": [], "layout": {"title": "No substances detected"}}

    pct_values = [round(float(v) * 100, 2) for v in present.values]

    bar_trace = {
        "type": "bar",
        "orientation": "h",
        "x": pct_values,
        "y": list(present.index),
        "text": [f"{v}%" for v in pct_values],
        "textposition": "outside",
        "marker": {"color": "#4a7fd4"},
        "hovertemplate": "<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    }

    layout = {
        "margin": {"l": 10, "r": 60, "t": 10, "b": 30},
        "xaxis": {
            "title": "Area %",
            "range": [0, max(pct_values) * 1.25],
        },
        "yaxis": {"automargin": True, "tickfont": {"size": 11}},
        "paper_bgcolor": "white",
        "plot_bgcolor": "#f8f9fa",
        "height": max(200, len(present) * 28 + 60),
    }

    return {"data": [bar_trace], "layout": layout}

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    datasets = get_datasets()
    return templates.TemplateResponse(
        request, "index.html", {"datasets": datasets}
    )


@app.post("/api/load")
async def load_dataset(req: LoadRequest):
    dataset_dir = DATA_DIR / req.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset}")

    try:
        fm = pd.read_csv(dataset_dir / "feature_matrix_wide_normalized.csv", index_col=0)
        fm.index = fm.index.astype(str)
        fm.columns = fm.columns.astype(str)

        cos = pd.read_csv(dataset_dir / "cosine_similarity.csv", index_col=0)
        jac = pd.read_csv(dataset_dir / "jaccard_similarity.csv", index_col=0)
        euc = pd.read_csv(dataset_dir / "euclidean_similarity.csv", index_col=0)
        for df in [cos, jac, euc]:
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)

        meta = pd.read_csv(dataset_dir / "sample_metadata.csv")
        sample_col = meta.columns[0]
        meta[sample_col] = meta[sample_col].astype(str)
        meta = meta.set_index(sample_col)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Compute TF-IDF and Pearson
    tfidf = TFIDFSimilarity(fm).similarity()
    pea = compute_pearson_sim(fm)

    # Extract sample years from metadata
    sample_years: Dict[str, int] = {}
    date_col = next(
        (c for c in meta.columns if "date" in c.lower() or "year" in c.lower()), None
    )
    if date_col:
        for sample_id, row in meta.iterrows():
            try:
                sample_years[str(sample_id)] = int(row[date_col])
            except (ValueError, TypeError):
                pass

    # Store in state
    state.dataset = req.dataset
    state.feature_matrix = fm
    state.metadata = meta
    state.cosine_sim = cos
    state.jaccard_sim = jac
    state.euclidean_sim = euc
    state.tfidf_sim = tfidf
    state.pearson_sim = pea
    state.sample_years = sample_years
    # Initialise active metrics to full dataset (no filters applied yet)
    state.active_cosine   = cos
    state.active_jaccard  = jac
    state.active_euclidean = euc
    state.active_tfidf    = tfidf
    state.active_pearson  = pea

    # Initial joint similarity + graph with default weights
    w = state.weights
    joint = compute_joint(
        cos, jac, euc, tfidf, pea,
        w["weight_cosine"], w["weight_jaccard"],
        w["weight_euclidean"], w["weight_tfidf"], w["weight_pearson"],
    )
    state.joint_sim = joint

    builder = FSISNetworkBuilder(joint, w["threshold"])
    G = builder.build_graph()
    state.graph = G

    figure = build_plotly_figure(G, joint)
    stats = graph_stats(G)

    year_vals = list(sample_years.values())
    year_range = (
        {"min": int(min(year_vals)), "max": int(max(year_vals))}
        if year_vals else None
    )

    return JSONResponse({"graph": figure, "stats": stats, "year_range": year_range})


@app.post("/api/compute")
async def compute(req: ComputeRequest):
    if state.cosine_sim is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    state.weights = {
        "weight_cosine": req.weight_cosine,
        "weight_jaccard": req.weight_jaccard,
        "weight_euclidean": req.weight_euclidean,
        "weight_tfidf": req.weight_tfidf,
        "weight_pearson": req.weight_pearson,
        "threshold": req.threshold,
    }
    state.sqrt_pretreatment = req.sqrt_pretreatment

    cos = state.cosine_sim
    jac = state.jaccard_sim
    euc = state.euclidean_sim
    tfidf_m = state.tfidf_sim
    pea_m = state.pearson_sim

    # Determine which samples to keep (chemical + year filters combined)
    fm = state.feature_matrix
    keep = set(fm.index)

    if req.filter_chemicals:
        if req.filter_mode == "all":
            mask = pd.Series(True, index=fm.index)
            for chem in req.filter_chemicals:
                if chem in fm.columns:
                    mask &= (fm[chem] > 0)
        else:
            mask = pd.Series(False, index=fm.index)
            for chem in req.filter_chemicals:
                if chem in fm.columns:
                    mask |= (fm[chem] > 0)
        keep &= set(fm.index[mask].tolist())

    if state.sample_years and (req.year_min is not None or req.year_max is not None):
        year_keep = {
            s for s, yr in state.sample_years.items()
            if (req.year_min is None or yr >= req.year_min)
            and (req.year_max is None or yr <= req.year_max)
        }
        keep &= year_keep

    keep_list = [s for s in fm.index if s in keep]

    if not keep_list:
        state.joint_sim = pd.DataFrame()
        G = nx.Graph()
        state.graph = G
        figure = build_plotly_figure(G, pd.DataFrame())
        stats = graph_stats(G)
        return JSONResponse({"graph": figure, "stats": stats})

    if req.sqrt_pretreatment:
        fm_work = apply_sqrt_pretreatment(fm.loc[keep_list])
        cos = compute_cosine_sim(fm_work)
        jac = compute_jaccard_sim(fm_work)
        euc = compute_euclidean_sim(fm_work)
        tfidf_m = TFIDFSimilarity(fm_work).similarity()
        pea_m = compute_pearson_sim(fm_work)
    else:
        cos = cos.loc[keep_list, keep_list]
        jac = jac.loc[keep_list, keep_list]
        euc = euc.loc[keep_list, keep_list]
        tfidf_m = tfidf_m.loc[keep_list, keep_list]
        pea_m = pea_m.loc[keep_list, keep_list]

    joint = compute_joint(
        cos, jac, euc, tfidf_m, pea_m,
        req.weight_cosine, req.weight_jaccard,
        req.weight_euclidean, req.weight_tfidf, req.weight_pearson,
    )
    state.joint_sim = joint
    # Store filtered individual metrics for analytics endpoints
    state.active_cosine   = cos
    state.active_jaccard  = jac
    state.active_euclidean = euc
    state.active_tfidf    = tfidf_m
    state.active_pearson  = pea_m

    builder = FSISNetworkBuilder(joint, req.threshold)
    G = builder.build_graph()
    state.graph = G

    figure = build_plotly_figure(G, joint)
    stats = graph_stats(G)

    return JSONResponse({"graph": figure, "stats": stats})


@app.get("/api/chemicals")
async def get_chemicals():
    if state.feature_matrix is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    fm = state.feature_matrix
    n_samples = len(fm)
    freq = (fm > 0).sum(axis=0)
    avg_conc = fm[fm > 0].mean(axis=0).fillna(0)  # mean only where present

    # Load forensic roles from substance_categories.csv
    substance_roles: Dict[str, str] = {}
    cat_csv = DATA_DIR / "substance_categories.csv"
    if cat_csv.exists():
        try:
            cat_df = pd.read_csv(cat_csv)
            for _, row in cat_df.iterrows():
                sub_name = str(row.get("Substance", "")).strip()
                role = str(row.get("Role", "")).strip()
                if sub_name and role:
                    substance_roles[sub_name] = role
        except Exception:
            pass

    chemicals = [
        {
            "name": sub,
            "count": int(freq[sub]),
            "pct": round(float(freq[sub]) / n_samples * 100, 1),
            "avg_pct": round(float(avg_conc[sub]), 2),
            "role": substance_roles.get(sub, "Unknown"),
        }
        for sub in freq.index
        if freq[sub] > 0
    ]
    chemicals.sort(key=lambda x: x["count"], reverse=True)

    return JSONResponse({"chemicals": chemicals, "n_samples": n_samples})


@app.get("/api/sample/{sample_name}")
async def sample_detail(sample_name: str):
    if state.feature_matrix is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    if sample_name not in state.feature_matrix.index:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")

    # Metadata
    meta_row: Dict[str, Any] = {}
    if state.metadata is not None and sample_name in state.metadata.index:
        row = state.metadata.loc[sample_name]
        meta_row = {str(k): (str(v) if pd.notna(v) else "") for k, v in row.items()}

    # Connections in current graph
    connections = 0
    if state.graph is not None and sample_name in state.graph:
        connections = state.graph.degree(sample_name)

    # Substance count
    fm_row = state.feature_matrix.loc[sample_name]
    substance_count = int((fm_row > 0).sum())

    # Bar chart
    chart = build_sample_chart(sample_name, state.feature_matrix)

    return JSONResponse({
        "name": sample_name,
        "metadata": meta_row,
        "connections": connections,
        "substance_count": substance_count,
        "chart": chart,
    })


@app.post("/api/selection-summary")
async def selection_summary(req: SelectionRequest):
    if state.feature_matrix is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    fm = state.feature_matrix
    samples = [s for s in req.samples if s in fm.index]

    if len(samples) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid samples.")

    sub_matrix = fm.loc[samples]
    present = sub_matrix > 0.0

    n = len(samples)

    # Count how many selected samples contain each substance
    counts = present.sum(axis=0)

    all_subs = counts[counts == n].index.tolist()
    some_subs = counts[(counts > 0) & (counts < n)].index.tolist()

    def substance_detail(sub: str) -> Dict[str, Any]:
        vals = []
        for s in samples:
            v = float(fm.loc[s, sub])
            if v > 0:
                vals.append({"sample": s, "value": round(v * 100, 2)})
        return {"name": sub, "values": vals}

    # Sort by average value across selected samples descending
    def avg_val(sub: str) -> float:
        return float(sub_matrix[sub].mean())

    all_detail = sorted([substance_detail(s) for s in all_subs], key=lambda x: avg_val(x["name"]), reverse=True)
    some_detail = sorted([substance_detail(s) for s in some_subs], key=lambda x: avg_val(x["name"]), reverse=True)

    # Notable absences: top substances by frequency in full dataset not in any selected sample
    full_freq = (fm > 0).sum(axis=0)
    absent_in_selection = counts[counts == 0].index
    top_absent = (
        full_freq[absent_in_selection]
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    absent_detail = [
        {"name": sub, "freq_in_dataset": int(full_freq[sub])}
        for sub in top_absent
    ]

    return JSONResponse({
        "selected": samples,
        "n": n,
        "all": all_detail,
        "some": some_detail,
        "notable_absences": absent_detail,
        "total_samples_in_dataset": len(fm),
    })


@app.get("/api/export/graphml")
async def export_graphml():
    if state.graph is None or state.graph.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No graph computed. Load a dataset and click Compute first.")

    try:
        G = state.graph.copy()

        degree_cent = nx.degree_centrality(G)
        try:
            between_cent = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            between_cent = {node: 0.0 for node in G.nodes()}
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=500)
        except Exception:
            eigen_cent = {node: 0.0 for node in G.nodes()}

        # GraphML cannot serialize dict values — strip graph-level dict attributes
        # (FSISNetworkBuilder stores G.graph["fsis"] = {...})
        for key in list(G.graph.keys()):
            if isinstance(G.graph[key], (dict, list)):
                del G.graph[key]
        G.graph["threshold"] = float(state.weights.get("threshold", 0.5))
        G.graph["dataset"] = state.dataset or ""

        for node in G.nodes():
            G.nodes[node]["degree"] = int(G.degree(node))
            G.nodes[node]["degree_centrality"] = round(float(degree_cent.get(node, 0.0)), 6)
            G.nodes[node]["betweenness_centrality"] = round(float(between_cent.get(node, 0.0)), 6)
            G.nodes[node]["eigenvector_centrality"] = round(float(eigen_cent.get(node, 0.0)), 6)
            if state.metadata is not None and node in state.metadata.index:
                for k, v in state.metadata.loc[node].items():
                    if pd.notna(v):
                        G.nodes[node][str(k)] = str(v)

        if state.joint_sim is not None:
            for u, v in G.edges():
                try:
                    val = float(state.joint_sim.loc[u, v])
                    if not np.isnan(val):
                        G[u][v]["joint_similarity"] = val
                except (KeyError, TypeError):
                    pass

        buf = io.BytesIO()
        nx.write_graphml(G, buf)
        content = buf.getvalue()

        dataset_name = (state.dataset or "graph").replace(" ", "_")
        return Response(
            content=content,
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{dataset_name}_fsis.graphml"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/api/network-analysis")
async def network_analysis_endpoint():
    if state.graph is None or state.graph.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No graph computed. Load a dataset and click Compute first.")

    G = state.graph
    n = G.number_of_nodes()

    # Centrality
    degree_cent = nx.degree_centrality(G)
    try:
        between_cent = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        between_cent = {node: 0.0 for node in G.nodes()}
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=500)
    except Exception:
        eigen_cent = {node: 0.0 for node in G.nodes()}

    def top_n(d: Dict, n_results: int = 15) -> List[Dict]:
        return [{"node": k, "value": round(v, 4)}
                for k, v in sorted(d.items(), key=lambda x: -x[1])[:n_results]]

    # Community detection — all four algorithms
    communities: Dict[str, Any] = {}

    try:
        lv = list(nx.community.louvain_communities(G, seed=42))
        communities["louvain"] = {
            "name": "Louvain",
            "description": "Optimises modularity iteratively. Good general-purpose algorithm.",
            "n_communities": len(lv),
            "modularity": round(nx.community.modularity(G, lv), 4),
            "communities": [sorted(list(c)) for c in sorted(lv, key=len, reverse=True)],
        }
    except Exception as exc:
        communities["louvain"] = {"name": "Louvain", "error": str(exc)}

    try:
        gm = list(nx.community.greedy_modularity_communities(G))
        communities["greedy_modularity"] = {
            "name": "Greedy Modularity",
            "description": "Agglomerative algorithm maximising modularity. Fast on large graphs.",
            "n_communities": len(gm),
            "modularity": round(nx.community.modularity(G, gm), 4),
            "communities": [sorted(list(c)) for c in sorted(gm, key=len, reverse=True)],
        }
    except Exception as exc:
        communities["greedy_modularity"] = {"name": "Greedy Modularity", "error": str(exc)}

    try:
        lp = list(nx.community.label_propagation_communities(G))
        communities["label_propagation"] = {
            "name": "Label Propagation",
            "description": "Fast stochastic algorithm. Results may vary slightly between runs.",
            "n_communities": len(lp),
            "modularity": round(nx.community.modularity(G, lp), 4),
            "communities": [sorted(list(c)) for c in sorted(lp, key=len, reverse=True)],
        }
    except Exception as exc:
        communities["label_propagation"] = {"name": "Label Propagation", "error": str(exc)}

    if n <= 200:
        try:
            gn = list(next(nx.community.girvan_newman(G)))
            communities["girvan_newman"] = {
                "name": "Girvan-Newman",
                "description": "Edge-betweenness based. Deterministic but slow — only runs on graphs ≤200 nodes.",
                "n_communities": len(gn),
                "modularity": round(nx.community.modularity(G, gn), 4),
                "communities": [sorted(list(c)) for c in sorted(gn, key=len, reverse=True)],
            }
        except Exception as exc:
            communities["girvan_newman"] = {"name": "Girvan-Newman", "error": str(exc)}
    else:
        communities["girvan_newman"] = {
            "name": "Girvan-Newman",
            "skipped": True,
            "reason": f"Skipped — graph has {n} nodes (limit: 200 for performance).",
        }

    return JSONResponse({
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "centrality": {
            "degree": top_n(degree_cent),
            "betweenness": top_n(between_cent),
            "eigenvector": top_n(eigen_cent),
        },
        "communities": communities,
    })


# ─────────────────────────────────────────────
# Analytics endpoints
# ─────────────────────────────────────────────

ANALYTICS_CONFIGS = [
    ("Cosine only",    {"w_cos": 1, "w_jac": 0, "w_euc": 0, "w_tfidf": 0, "w_pea": 0}),
    ("Jaccard only",   {"w_cos": 0, "w_jac": 1, "w_euc": 0, "w_tfidf": 0, "w_pea": 0}),
    ("Euclidean only", {"w_cos": 0, "w_jac": 0, "w_euc": 1, "w_tfidf": 0, "w_pea": 0}),
    ("TF-IDF only",    {"w_cos": 0, "w_jac": 0, "w_euc": 0, "w_tfidf": 1, "w_pea": 0}),
    ("Pearson only",   {"w_cos": 0, "w_jac": 0, "w_euc": 0, "w_tfidf": 0, "w_pea": 1}),
    ("Joint (equal)",  {"w_cos": 1, "w_jac": 1, "w_euc": 1, "w_tfidf": 1, "w_pea": 1}),
]

SWEEP_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _require_active_metrics():
    """Raise if analytics cannot run."""
    if state.active_cosine is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Load a dataset and click Compute first.")


def _graph_stats_dict(G: nx.Graph, n_total: int) -> Dict:
    comps    = list(nx.connected_components(G))
    iso      = sum(1 for c in comps if len(c) == 1)
    clusters = [c for c in comps if len(c) > 1]
    largest  = max((len(c) for c in clusters), default=0)
    return {
        "edges":    G.number_of_edges(),
        "clusters": len(clusters),
        "isolates": iso,
        "largest":  largest,
        "largest_pct": round(100 * largest / n_total, 1) if n_total else 0,
    }


def _joint_for_config(w: Dict) -> pd.DataFrame:
    return compute_joint(
        state.active_cosine, state.active_jaccard,
        state.active_euclidean, state.active_tfidf, state.active_pearson,
        w["w_cos"], w["w_jac"], w["w_euc"], w["w_tfidf"], w["w_pea"],
    )


@app.get("/api/analytics/metric-comparison")
async def analytics_metric_comparison():
    _require_active_metrics()
    threshold = state.weights.get("threshold", 0.5)
    n_total   = len(state.active_cosine)
    rows = []
    for name, w in ANALYTICS_CONFIGS:
        jt = _joint_for_config(w)
        G  = FSISNetworkBuilder(jt, threshold).build_graph()
        st = _graph_stats_dict(G, n_total)
        rows.append({"metric": name, **st})
    return JSONResponse({
        "threshold": threshold,
        "n_samples": n_total,
        "rows": rows,
    })


@app.get("/api/analytics/threshold-sweep")
async def analytics_threshold_sweep():
    _require_active_metrics()
    n_total = len(state.active_cosine)
    w       = state.weights

    # Current user weights
    jt_user = compute_joint(
        state.active_cosine, state.active_jaccard,
        state.active_euclidean, state.active_tfidf, state.active_pearson,
        w["weight_cosine"], w["weight_jaccard"],
        w["weight_euclidean"], w["weight_tfidf"], w["weight_pearson"],
    )
    # Equal-weight joint
    jt_equal = _joint_for_config({"w_cos": 1, "w_jac": 1, "w_euc": 1, "w_tfidf": 1, "w_pea": 1})

    results_user  = []
    results_equal = []
    for t in SWEEP_THRESHOLDS:
        for jt, out in [(jt_user, results_user), (jt_equal, results_equal)]:
            G  = FSISNetworkBuilder(jt, t).build_graph()
            st = _graph_stats_dict(G, n_total)
            out.append({"threshold": t, **st})

    return JSONResponse({
        "thresholds":    SWEEP_THRESHOLDS,
        "n_samples":     n_total,
        "current_weights": results_user,
        "equal_weights":   results_equal,
    })


@app.get("/api/analytics/cluster-profiles")
async def analytics_cluster_profiles():
    if state.graph is None or state.graph.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No graph computed.")
    if state.feature_matrix is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    G    = state.graph
    fm   = state.feature_matrix
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    clusters = []
    for i, comp in enumerate(comps):
        members = sorted(comp)
        is_iso  = len(comp) == 1
        valid   = [m for m in members if m in fm.index]
        if not valid:
            continue

        sub_fm = fm.loc[valid]
        means  = sub_fm.mean()
        top    = means[means > 0.001].nlargest(8)

        # Within-cluster mean similarity
        within_sim = None
        if state.joint_sim is not None and len(valid) >= 2:
            avail = [m for m in valid if m in state.joint_sim.index]
            if len(avail) >= 2:
                arr = state.joint_sim.loc[avail, avail].to_numpy(dtype=np.float64)
                idx = np.triu_indices(len(arr), k=1)
                if len(idx[0]) > 0:
                    within_sim = round(float(arr[idx].mean()), 4)

        # Year distribution
        year_dist: Dict[str, int] = {}
        if state.sample_years:
            for m in members:
                yr = state.sample_years.get(m)
                if yr is not None:
                    year_dist[str(yr)] = year_dist.get(str(yr), 0) + 1

        clusters.append({
            "cluster_id":  i + 1,
            "size":        len(members),
            "is_isolate":  is_iso,
            "members":     members,
            "within_sim":  within_sim,
            "year_dist":   year_dist,
            "top_substances": [
                {"name": str(k), "mean_pct": round(float(v) * 100, 2)}
                for k, v in top.items()
            ],
        })

    n_clusters = sum(1 for c in clusters if not c["is_isolate"])
    n_isolates = sum(1 for c in clusters if c["is_isolate"])

    return JSONResponse({
        "n_clusters": n_clusters,
        "n_isolates": n_isolates,
        "clusters":   clusters,
    })


@app.get("/api/analytics/temporal-split")
async def analytics_temporal_split():
    _require_active_metrics()
    if not state.sample_years:
        raise HTTPException(status_code=400, detail="No year data available for this dataset.")

    # Determine year range from active samples
    active_ids = list(state.active_cosine.index)
    active_years = {s: state.sample_years[s] for s in active_ids if s in state.sample_years}
    if not active_years:
        raise HTTPException(status_code=400, detail="No year data for current active samples.")

    all_years = sorted(set(active_years.values()))
    n_total   = len(active_ids)

    # Per-year statistics
    year_stats = []
    for yr in all_years:
        members = [s for s, y in active_years.items() if y == yr]
        if len(members) < 2:
            year_stats.append({
                "year": yr, "n": len(members),
                "mean_sim": None, "std_sim": None,
                "clusters": None, "isolates": None,
            })
            continue
        avail = [m for m in members if m in state.joint_sim.index]
        if len(avail) < 2:
            continue
        arr  = state.joint_sim.loc[avail, avail].to_numpy(dtype=np.float64)
        idx  = np.triu_indices(len(arr), k=1)
        vals = arr[idx]
        G_yr = FSISNetworkBuilder(
            state.joint_sim.loc[avail, avail],
            state.weights.get("threshold", 0.5)
        ).build_graph()
        st_yr = _graph_stats_dict(G_yr, len(avail))
        year_stats.append({
            "year":      yr,
            "n":         len(members),
            "mean_sim":  round(float(vals.mean()), 4),
            "std_sim":   round(float(vals.std()), 4),
            "clusters":  st_yr["clusters"],
            "isolates":  st_yr["isolates"],
            "largest":   st_yr["largest"],
            "largest_pct": st_yr["largest_pct"],
        })

    # Cross-year mean similarities (pairwise between years)
    cross_year = []
    for i, yr_a in enumerate(all_years):
        for yr_b in all_years[i + 1:]:
            ids_a = [s for s, y in active_years.items() if y == yr_a]
            ids_b = [s for s, y in active_years.items() if y == yr_b]
            av_a  = [s for s in ids_a if s in state.joint_sim.index]
            av_b  = [s for s in ids_b if s in state.joint_sim.index]
            if not av_a or not av_b:
                continue
            arr = state.joint_sim.loc[av_a, av_b].to_numpy(dtype=np.float64)
            cross_year.append({
                "year_a": yr_a,
                "year_b": yr_b,
                "mean_sim": round(float(arr.mean()), 4),
                "n_pairs":  arr.size,
            })

    # Top substances per year (for composition timeline)
    year_composition = {}
    if state.feature_matrix is not None:
        fm = state.feature_matrix
        for yr in all_years:
            members = [s for s, y in active_years.items() if y == yr]
            valid   = [m for m in members if m in fm.index]
            if valid:
                top = fm.loc[valid].mean().nlargest(6)
                year_composition[str(yr)] = [
                    {"name": str(k), "mean_pct": round(float(v) * 100, 2)}
                    for k, v in top.items() if v > 0.001
                ]

    return JSONResponse({
        "all_years":        all_years,
        "year_stats":       year_stats,
        "cross_year":       cross_year,
        "year_composition": year_composition,
    })
