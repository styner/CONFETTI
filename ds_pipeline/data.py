"""Data loading: VTKs, adjacency matrices, labels CSV, per-subject tensors.

Public API:
  load_levels()           -- read VTKs + sparse adjacencies for every level
  load_v06_subjects()     -- read the V06 label CSV, return DataFrame
  pool_maps_from_levels() -- precompute L_i -> L_{i+1} index maps
  build_subject_tensors() -- assemble per-subject (level, n_nodes, n_feat) tensors

The merged-imputed VTK shape (from build_neighborhood_graph.py per-level output)
is identical at every level except for the node count. Each VTK carries:
  - POINTS  (x, y, z)
  - point-data 'SamplingDistance2Origin' (arclength)
  - point-data 'FiberLabel' (tract id)
  - point-data '<property>_<subject>' for each (property, subject)
  - cell-data 'FiberLabel' (one per polyline cell, not used here)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .config import (
    COHORT_COLUMN,
    COHORT_TO_LABEL,
    COVARIATE_CSV_COLUMNS,
    COVARIATE_NAMES,
    DEFAULT_LEVEL_BASE,
    MISSING_VALUE,
    PROPERTIES,
    SEX_MAP,
    VINELAND_COLUMNS_V06,
    VINELAND_SHORT_NAMES,
    level_indices_path,
    level_npz_path,
    level_vtk_path,
)


# ---------------------------------------------------------------------------
# Per-level container
# ---------------------------------------------------------------------------


@dataclass
class LevelData:
    """All graph-level data shared across subjects at one resolution."""
    level: int
    coords: np.ndarray         # (N, 3) float32
    arclength: np.ndarray      # (N,)   float32
    fiber_label: np.ndarray    # (N,)   int32
    kept_orig: np.ndarray      # (N,)   int64 -- index back to L0
    adj: sp.csr_matrix         # (N, N) sparse 0/1 adjacency
    edge_index: np.ndarray     # (2, E) directed edges in COO form (for PyG)
    edge_weight: np.ndarray    # (E,)  -- exp(-d^2/sigma^2)
    # For each subject, the raw per-property numpy arrays (with -1 sentinels).
    property_values: dict[str, dict[str, np.ndarray]]  # property -> subject -> (N,)


# ---------------------------------------------------------------------------
# VTK reading
# ---------------------------------------------------------------------------


def _read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    pd = reader.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read polydata or empty file: {path}")
    return pd


def _parse_property_arrays(
    polydata: vtk.vtkPolyData, properties: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    """Pull per-(property, subject) arrays out of a polydata's PointData.
    Returns property -> subject -> 1D numpy array."""
    out: dict[str, dict[str, np.ndarray]] = {p: {} for p in properties}
    prop_set = set(properties)
    pdata = polydata.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        name = pdata.GetArrayName(i)
        if not name or "_" not in name:
            continue
        prefix, rest = name.split("_", 1)
        if prefix not in prop_set:
            continue
        out[prefix][rest] = vtk_to_numpy(pdata.GetArray(name)).astype(np.float32)
    return out


def _adjacency_to_edge_index(
    adj: sp.csr_matrix, coords: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 0/1 sparse adjacency to PyG-style (edge_index, edge_weight)
    where edge weights are exp(-d^2/sigma^2). Edges are duplicated (i,j) and
    (j,i) for PyG's directed convention."""
    coo = adj.tocoo()
    src = coo.row.astype(np.int64)
    dst = coo.col.astype(np.int64)
    edge_index = np.vstack([src, dst])
    delta = coords[src] - coords[dst]
    d2 = np.einsum("ij,ij->i", delta, delta)
    edge_weight = np.exp(-d2 / max(sigma * sigma, 1e-9)).astype(np.float32)
    return edge_index, edge_weight


def load_levels(
    base_dir: Path,
    levels: tuple[int, ...],
    base: str = DEFAULT_LEVEL_BASE,
    properties: tuple[str, ...] = PROPERTIES,
    edge_sigma_factor: float = 1.0,
) -> dict[int, LevelData]:
    """Read VTKs and sparse adjacencies for every level.

    `edge_sigma_factor` scales the median nearest-cross-tract distance (the
    threshold used to build the graph) to set the Gaussian edge-weight sigma.
    A larger sigma -> softer falloff.
    """
    out: dict[int, LevelData] = {}
    for L in levels:
        vtk_path = level_vtk_path(base_dir, L, base)
        npz_path = level_npz_path(base_dir, L, base)
        idx_path = level_indices_path(base_dir, L, base)
        if not (vtk_path.is_file() and npz_path.is_file() and idx_path.is_file()):
            raise SystemExit(
                f"Level {L} files missing under {base_dir}: expected "
                f"{vtk_path.name}, {npz_path.name}, {idx_path.name}"
            )
        polydata = _read_polydata(vtk_path)
        coords = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
        arclength = vtk_to_numpy(
            polydata.GetPointData().GetArray("SamplingDistance2Origin")
        ).astype(np.float32)
        fiber_label = vtk_to_numpy(
            polydata.GetPointData().GetArray("FiberLabel")
        ).astype(np.int32)
        prop_arrays = _parse_property_arrays(polydata, properties)
        kept_orig = np.loadtxt(idx_path, dtype=np.int64)
        adj = sp.load_npz(npz_path).astype(np.uint8).tocsr()
        # Sigma: half the distance threshold (a soft fall-off across edges).
        # Approx the median nearest-cross-tract distance via Euclidean stats
        # of the connected pairs.
        coo = adj.tocoo()
        if coo.nnz:
            sample = coo.data.size
            delta = coords[coo.row] - coords[coo.col]
            d = np.linalg.norm(delta, axis=1)
            sigma = float(np.median(d)) * edge_sigma_factor
        else:
            sigma = 1.0
        edge_index, edge_weight = _adjacency_to_edge_index(adj, coords, sigma)
        out[L] = LevelData(
            level=L,
            coords=coords,
            arclength=arclength,
            fiber_label=fiber_label,
            kept_orig=kept_orig,
            adj=adj,
            edge_index=edge_index,
            edge_weight=edge_weight,
            property_values=prop_arrays,
        )
    return out


def pool_maps_from_levels(
    levels_data: dict[int, LevelData],
) -> dict[tuple[int, int], np.ndarray]:
    """Build pool maps M[(L, L+1)]: for each new index at L+1, the row in
    `kept_orig` at level L that corresponds to it. Used by Graph U-Net to
    pool/unpool features across levels.

    Strict nesting (kept_Lp1 ⊂ kept_L) holds in most cases because the level
    builder samples every 2**L-th point along each polyline. However, the
    "minimum 3 points per fiber" padding can pick different midpoints at
    adjacent depths (e.g. a length-6 polyline keeps {0,4,5} at L2 but {0,2,5}
    at L3 -- the midpoint differs). When that happens, the L+1 node whose L0
    id is missing from kept_L falls back to its **spatially nearest** L node
    (3D Euclidean), so unet pool/unpool stays valid without regenerating the
    graph files.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    sorted_levels = sorted(levels_data)
    for L, Lp1 in zip(sorted_levels[:-1], sorted_levels[1:]):
        kept_L = levels_data[L].kept_orig
        kept_Lp1 = levels_data[Lp1].kept_orig
        coords_L = levels_data[L].coords
        coords_Lp1 = levels_data[Lp1].coords
        l0_to_L = {int(idx): pos for pos, idx in enumerate(kept_L)}

        mp = np.empty(len(kept_Lp1), dtype=np.int64)
        n_fallback = 0
        tree = None
        for i, l0_idx in enumerate(kept_Lp1):
            pos = l0_to_L.get(int(l0_idx))
            if pos is not None:
                mp[i] = pos
            else:
                if tree is None:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(coords_L)
                _, nn_pos = tree.query(coords_Lp1[i], k=1)
                mp[i] = int(nn_pos)
                n_fallback += 1
        if n_fallback:
            print(
                f"  pool map L{L} -> L{Lp1}: "
                f"{n_fallback}/{len(kept_Lp1)} nodes used nearest-neighbor "
                f"fallback (min-3 padding nesting break)"
            )
        out[(L, Lp1)] = mp
    return out


# ---------------------------------------------------------------------------
# Label CSV
# ---------------------------------------------------------------------------


def load_v06_subjects(
    csv_path: Path,
    cohort_column: str = COHORT_COLUMN,
    cohort_map: dict[str, int] | None = None,
    vineland_columns: tuple[str, ...] = VINELAND_COLUMNS_V06,
    vineland_short: tuple[str, ...] = VINELAND_SHORT_NAMES,
    covariate_columns: tuple[str, ...] = COVARIATE_CSV_COLUMNS,
    covariate_names: tuple[str, ...] = COVARIATE_NAMES,
    extra_target_columns: tuple[str, ...] = (),
    extra_target_short_names: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load V06 row, parse subject id, cohort label, Vineland scores, and the
    head-concatenated covariates.

    Returns a DataFrame indexed by 'subject_id' ('<CandID>_V06') with columns:
      cohort           -- original string label
      label            -- 0/1 (Control/DS) per COHORT_TO_LABEL; NaN if unrecognized
      vineland_<short> -- one column per V06 Vineland subscale (NaN if missing)
      target_<short>   -- one column per extra regression target (typically the
                          V24 Vineland + Bayley-4 outcomes for prospective
                          regression); the short names need to be unique across
                          target families. Numeric, NaN where missing.
      cov_<name>       -- one column per covariate; sex is recoded F=0/M=1; the
                          other covariates are float, NaN where missing.
    """
    cohort_map = cohort_map or COHORT_TO_LABEL
    df = pd.read_csv(csv_path)
    cand_id = df["CandID"].astype(str)
    visit = df["Visit"].astype(str)
    subject_id = (cand_id + "_" + visit).tolist()
    out = pd.DataFrame(index=subject_id)
    out.index.name = "subject_id"
    out["cohort"] = df[cohort_column].astype(str).values
    out["label"] = out["cohort"].map(cohort_map)
    for short, full in zip(vineland_short, vineland_columns):
        col = pd.to_numeric(df[full], errors="coerce")
        out[f"vineland_{short}"] = col.values
    if len(extra_target_columns) != len(extra_target_short_names):
        raise ValueError(
            "extra_target_columns and extra_target_short_names must be the same length"
        )
    for short, full in zip(extra_target_short_names, extra_target_columns):
        col = pd.to_numeric(df[full], errors="coerce")
        out[f"target_{short}"] = col.values

    # Covariates. Sex is categorical -> numeric via SEX_MAP; the others are
    # parsed as floats; non-numeric / empty cells become NaN and are
    # mean-imputed per training fold downstream.
    for raw, name in zip(covariate_columns, covariate_names):
        raw_vals = df[raw]
        if name == "sex":
            out[f"cov_{name}"] = raw_vals.map(SEX_MAP).astype(float).values
        else:
            out[f"cov_{name}"] = pd.to_numeric(raw_vals, errors="coerce").values
    return out


def build_covariate_matrix(
    label_df: pd.DataFrame,
    subject_ids: list[str],
    covariate_names: tuple[str, ...] = COVARIATE_NAMES,
) -> np.ndarray:
    """Stack the per-subject covariate columns into a (S, n_cov) float matrix.
    Missing entries stay as NaN -- training-fold mean imputation happens in cv.
    """
    cols = [f"cov_{name}" for name in covariate_names]
    return label_df.loc[subject_ids, cols].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-subject tensor assembly
# ---------------------------------------------------------------------------


def select_subjects(
    levels_data: dict[int, LevelData],
    label_df: pd.DataFrame,
    require_label: bool = True,
    require_any_vineland: bool = False,
) -> list[str]:
    """Subjects that (a) appear in the merged VTK (have property arrays) and
    (b) optionally have a non-NaN cohort label or any non-NaN Vineland score."""
    vtk_subjects: set[str] = set()
    any_level = next(iter(levels_data.values()))
    for prop in any_level.property_values:
        vtk_subjects.update(any_level.property_values[prop].keys())
    keep: list[str] = []
    vineland_cols = [c for c in label_df.columns if c.startswith("vineland_")]
    for sid in label_df.index:
        if sid not in vtk_subjects:
            continue
        if require_label and pd.isna(label_df.loc[sid, "label"]):
            continue
        if require_any_vineland and label_df.loc[sid, vineland_cols].isna().all():
            continue
        keep.append(sid)
    return keep


def build_subject_tensor_for_level(
    level_data: LevelData,
    subject_ids: list[str],
    imputed_values: dict[str, dict[str, np.ndarray]] | None = None,
    properties: tuple[str, ...] = PROPERTIES,
) -> np.ndarray:
    """Stack per-subject node-feature tensors for one level.

    Returns shape (S, N, F) where S = number of subjects, N = nodes at this
    level, and F = len(properties) + 4 spatial features (x, y, z, arclength).

    If `imputed_values` is supplied it replaces `level_data.property_values`
    for the lookup; useful when imputation is performed per-fold.
    """
    n = level_data.coords.shape[0]
    n_prop = len(properties)
    n_feat = n_prop + 4
    src = imputed_values if imputed_values is not None else level_data.property_values
    spatial = np.empty((n, 4), dtype=np.float32)
    spatial[:, :3] = level_data.coords
    spatial[:, 3] = level_data.arclength
    out = np.empty((len(subject_ids), n, n_feat), dtype=np.float32)
    for s_idx, sid in enumerate(subject_ids):
        for p_idx, prop in enumerate(properties):
            arr = src[prop].get(sid)
            if arr is None:
                arr = np.full(n, MISSING_VALUE, dtype=np.float32)
            out[s_idx, :, p_idx] = arr
        out[s_idx, :, n_prop:] = spatial
    return out


def coordinate_bounds(level_data: LevelData) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, half-range) for the (x, y, z, arclength) spatial feats."""
    coords4 = np.empty((level_data.coords.shape[0], 4), dtype=np.float32)
    coords4[:, :3] = level_data.coords
    coords4[:, 3] = level_data.arclength
    lo = coords4.min(axis=0)
    hi = coords4.max(axis=0)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    half = np.where(half > 0, half, 1.0)
    return center.astype(np.float32), half.astype(np.float32)
