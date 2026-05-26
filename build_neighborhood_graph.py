#!/usr/bin/env python3
"""Build a neighborhood adjacency graph over imputed fiber-axis points.

A pair of axis points is a neighbor in the graph if EITHER:
  - their 3D distance is below a threshold, OR
  - they are immediate predecessor/successor points along the same axis polyline.

The threshold is computed as `2 × median(nearest_cross_tract_distance)`, where
the nearest cross-tract distance for each axis point is taken from the dense
full-fiber representation `Fibers_merged.vtk` (restricted to fiber points at
the same arclength as the axis point, drawn from a *different* tract).

Rationale: the axes are spatially sparse (one polyline per tract), so the
axis-to-axis nearest distance massively overestimates the local anatomical
neighborhood scale. Using the full fiber cloud at matched arclengths gives a
representative inter-tract proximity at each anatomical position.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy


def read_arrays(
    path: Path, label_name: str, arclength_name: str
) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, np.ndarray]:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read or empty polydata: {path}")
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    pdata = polydata.GetPointData()
    labels_arr = pdata.GetArray(label_name)
    arc_arr = pdata.GetArray(arclength_name)
    if labels_arr is None or arc_arr is None:
        raise SystemExit(
            f"Required arrays {label_name!r} or {arclength_name!r} missing in {path}"
        )
    return (
        polydata,
        points,
        vtk_to_numpy(labels_arr).astype(np.int64),
        vtk_to_numpy(arc_arr).astype(np.float64),
    )


def cross_tract_nearest_distances(
    axis_points: np.ndarray,
    axis_labels: np.ndarray,
    axis_arc: np.ndarray,
    fiber_points: np.ndarray,
    fiber_labels: np.ndarray,
    fiber_arc: np.ndarray,
    arc_tolerance: float,
    k_start: int,
) -> np.ndarray:
    """For each axis point, find the distance to its nearest fiber point that
    (a) is in a different tract and (b) has arclength within ±arc_tolerance of
    the axis point's arclength. Returns np.inf where no such neighbor exists."""
    n_axis = axis_points.shape[0]
    print(f"Building KDTree over {fiber_points.shape[0]:,} fiber points...")
    t0 = time.time()
    tree = cKDTree(fiber_points)
    print(f"  built in {time.time() - t0:.1f}s")

    nearest = np.full(n_axis, np.inf, dtype=np.float64)
    k = min(max(2, k_start), tree.n)

    print(f"Vectorized k-NN query (k={k}) on all {n_axis:,} axis points...")
    t0 = time.time()
    dists, idxs = tree.query(axis_points, k=k)
    print(f"  done in {time.time() - t0:.1f}s")

    matches = (fiber_labels[idxs] != axis_labels[:, None]) & (
        np.abs(fiber_arc[idxs] - axis_arc[:, None]) <= arc_tolerance
    )
    has_match = matches.any(axis=1)
    first_match = np.argmax(matches, axis=1)
    nearest[has_match] = dists[has_match, first_match[has_match]]

    missing_idx = np.where(~has_match)[0]
    if missing_idx.size:
        print(
            f"  {missing_idx.size} axis points missed; falling back to "
            f"per-point queries with doubled k..."
        )
        for i in missing_idx:
            ki = min(k * 2, tree.n)
            while ki <= tree.n:
                d_i, idx_i = tree.query(axis_points[i], k=ki)
                m = (fiber_labels[idx_i] != axis_labels[i]) & (
                    np.abs(fiber_arc[idx_i] - axis_arc[i]) <= arc_tolerance
                )
                if m.any():
                    nearest[i] = d_i[m][0]
                    break
                if ki == tree.n:
                    break
                ki = min(ki * 2, tree.n)

    return nearest


def collect_sequential_edges(polydata: vtk.vtkPolyData) -> np.ndarray:
    """Return an (M, 2) array of (a, b) with a < b for every consecutive pair
    of points within each polyline cell."""
    pairs: list[tuple[int, int]] = []
    for c in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(c)
        n = cell.GetNumberOfPoints()
        ids = [cell.GetPointId(k) for k in range(n)]
        for k in range(n - 1):
            a, b = ids[k], ids[k + 1]
            pairs.append((min(a, b), max(a, b)))
    return (
        np.asarray(pairs, dtype=np.int64)
        if pairs
        else np.zeros((0, 2), dtype=np.int64)
    )


def build_adjacency(
    axis_points: np.ndarray,
    axis_labels: np.ndarray,
    polydata: vtk.vtkPolyData,
    threshold: float,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    n = axis_points.shape[0]
    print(f"Ball-query: candidates within {threshold:.3f} mm...")
    t0 = time.time()
    tree = cKDTree(axis_points)
    candidates = tree.query_ball_point(axis_points, r=threshold)
    n_candidate_pairs = sum(
        sum(1 for j in cands if j > i) for i, cands in enumerate(candidates)
    )
    print(
        f"  {n_candidate_pairs:,} candidate undirected pairs in "
        f"{time.time() - t0:.2f}s"
    )

    # Per-source filter: from each axis point, keep at most one edge per other
    # tract -- the closest point in that tract.
    print("Selecting one closest neighbor per (source point, target tract)...")
    t0 = time.time()
    selected: set[tuple[int, int]] = set()
    for i, cands in enumerate(candidates):
        if not cands:
            continue
        js = np.asarray(cands, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        target_tracts = axis_labels[js]
        deltas = axis_points[js] - axis_points[i]
        d2 = np.einsum("ij,ij->i", deltas, deltas)
        # For each target tract, pick the j with the smallest distance.
        order = np.argsort(d2, kind="stable")
        sorted_js = js[order]
        sorted_tracts = target_tracts[order]
        seen: set[int] = set()
        for j, t in zip(sorted_js, sorted_tracts):
            if t in seen:
                continue
            seen.add(int(t))
            a, b = (i, int(j)) if i < j else (int(j), i)
            selected.add((a, b))
    print(
        f"  {len(selected):,} edges retained after per-tract selection "
        f"in {time.time() - t0:.2f}s"
    )

    seq_pairs = collect_sequential_edges(polydata)
    seq_set = set(map(tuple, seq_pairs)) if seq_pairs.size else set()
    extra_seq = seq_set - selected
    print(
        f"  {seq_pairs.shape[0]:,} sequential edges along polylines "
        f"({len(extra_seq):,} not already retained)"
    )

    union = selected | seq_set
    all_pairs = (
        np.array(sorted(union), dtype=np.int64)
        if union
        else np.zeros((0, 2), dtype=np.int64)
    )

    rows = np.concatenate([all_pairs[:, 0], all_pairs[:, 1]])
    cols = np.concatenate([all_pairs[:, 1], all_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1
    stats = {
        "candidate_pairs": int(n_candidate_pairs),
        "selected_edges": int(len(selected)),
        "sequential_edges": int(seq_pairs.shape[0]),
        "sequential_extra": int(len(extra_seq)),
        "unique_edges": int(all_pairs.shape[0]),
    }
    return adj, stats


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a neighborhood adjacency graph over imputed fiber-axis points "
            "using a distance threshold derived from the dense full-fiber "
            "representation."
        )
    )
    parser.add_argument(
        "axis",
        type=Path,
        help="Path to merged axis VTK (e.g. FiberAxisProfiles_merged_imputed.vtk).",
    )
    parser.add_argument(
        "--fibers",
        type=Path,
        default=Path("Fibers_merged.vtk"),
        help="Path to full-fiber merged VTK (default: Fibers_merged.vtk).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output sparse adjacency matrix (.npz). Defaults to "
            "<axis-stem>_neighborhood.npz next to the axis file."
        ),
    )
    parser.add_argument(
        "--arc-tolerance",
        type=float,
        default=0.5,
        help=(
            "Maximum |arclength difference| (mm) between an axis point and a "
            "fiber point when computing the nearest cross-tract distance "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=500,
        help=(
            "Initial k for the vectorized fiber-tree k-NN query (default: 500). "
            "Increase if many axis points report no cross-tract neighbor at k."
        ),
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.5,
        help=(
            "Multiplier applied to the median nearest cross-tract distance "
            "to derive the neighbor distance threshold (default: 1.5)."
        ),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help=(
            "Output dense adjacency text matrix path. Defaults to the "
            ".npz path with a .txt extension. Each row is space-separated "
            "0/1 values written via numpy.savetxt."
        ),
    )
    parser.add_argument(
        "--label-array",
        default="FiberLabel",
        help="Per-point tract-label array name (default: FiberLabel).",
    )
    parser.add_argument(
        "--arclength-array",
        default="SamplingDistance2Origin",
        help="Per-point arclength array name (default: SamplingDistance2Origin).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.axis.is_file():
        raise SystemExit(f"Axis file not found: {args.axis}")
    if not args.fibers.is_file():
        raise SystemExit(f"Fibers file not found: {args.fibers}")

    out_npz = args.output or args.axis.with_name(
        f"{args.axis.stem}_neighborhood.npz"
    )
    info_path = out_npz.with_suffix(".info.txt")

    print(f"Axis: {args.axis}")
    axis_pd, axis_pts, axis_lbl, axis_arc = read_arrays(
        args.axis, args.label_array, args.arclength_array
    )
    print(
        f"  {len(axis_pts):,} points, "
        f"{len(np.unique(axis_lbl))} tracts, "
        f"{axis_pd.GetNumberOfCells()} polyline cells"
    )

    print(f"Fibers: {args.fibers}")
    _, fb_pts, fb_lbl, fb_arc = read_arrays(
        args.fibers, args.label_array, args.arclength_array
    )
    print(
        f"  {len(fb_pts):,} points, "
        f"{len(np.unique(fb_lbl))} tracts"
    )

    nearest = cross_tract_nearest_distances(
        axis_pts,
        axis_lbl,
        axis_arc,
        fb_pts,
        fb_lbl,
        fb_arc,
        arc_tolerance=args.arc_tolerance,
        k_start=args.k_neighbors,
    )

    finite = nearest[np.isfinite(nearest)]
    n_failed = int(np.isinf(nearest).sum())
    if not finite.size:
        raise SystemExit("All axis points failed the cross-tract distance query.")

    median_nd = float(np.median(finite))
    threshold = args.threshold_multiplier * median_nd
    print()
    print(
        f"Nearest cross-tract distance over {finite.size:,} axis points "
        f"({n_failed} missing):"
    )
    print(
        f"  min={finite.min():.3f}  p10={np.quantile(finite, 0.1):.3f}  "
        f"median={median_nd:.3f}  p90={np.quantile(finite, 0.9):.3f}  "
        f"max={finite.max():.3f}"
    )
    print(
        f"Distance threshold = {args.threshold_multiplier} × median "
        f"= {threshold:.3f} mm"
    )
    print()

    adj, stats = build_adjacency(axis_pts, axis_lbl, axis_pd, threshold)
    avg_deg = adj.nnz / adj.shape[0] if adj.shape[0] else 0
    print(
        f"Adjacency: {adj.shape[0]} nodes, "
        f"{adj.nnz:,} non-zeros, "
        f"{stats['unique_edges']:,} unique undirected edges, "
        f"average degree {avg_deg:.1f}"
    )

    sp.save_npz(out_npz, adj)
    print(f"Saved sparse adjacency: {out_npz}")

    text_path = args.text_output or out_npz.with_suffix(".txt")
    np.savetxt(text_path, adj.toarray().astype(np.int8), fmt="%d")
    print(f"Saved dense text adjacency: {text_path}")

    with info_path.open("w") as f:
        f.write(f"Input axis: {args.axis}\n")
        f.write(f"Input fibers: {args.fibers}\n")
        f.write(f"Axis points: {len(axis_pts)}\n")
        f.write(f"Fiber points: {len(fb_pts)}\n")
        f.write(f"Tracts: {len(np.unique(axis_lbl))}\n")
        f.write(f"Arclength tolerance: {args.arc_tolerance}\n")
        f.write(f"Initial k for cross-tract query: {args.k_neighbors}\n")
        f.write(f"Axis points without cross-tract neighbor: {n_failed}\n")
        f.write(f"Median nearest cross-tract distance: {median_nd:.6f}\n")
        f.write(f"Threshold multiplier: {args.threshold_multiplier}\n")
        f.write(f"Distance threshold: {threshold:.6f}\n")
        f.write(f"Candidate pairs within threshold: {stats['candidate_pairs']}\n")
        f.write(
            f"Edges after per-tract closest-only selection: "
            f"{stats['selected_edges']}\n"
        )
        f.write(f"Sequential edges along axes (candidates): {stats['sequential_edges']}\n")
        f.write(
            f"Sequential edges not already retained: {stats['sequential_extra']}\n"
        )
        f.write(f"Unique undirected edges (union): {stats['unique_edges']}\n")
        f.write(f"Average degree: {avg_deg:.4f}\n")
    print(f"Summary: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
