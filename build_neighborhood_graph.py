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
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


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


def select_subsample_indices(
    n_pts: int, step: int, min_pts: int = 3
) -> list[int]:
    """Pick local indices into a polyline of `n_pts` points subject to:
      - Index 0 (first) is always included.
      - Index n_pts - 1 (last) is always included when n_pts >= 2.
      - Otherwise every `step`-th index from the start is kept.
      - The result has at least `min_pts` indices when `n_pts >= min_pts`;
        gaps are filled by inserting midpoints into the largest remaining gap.
    """
    if n_pts <= 0:
        return []
    if n_pts <= min_pts:
        return list(range(n_pts))

    kept = list(range(0, n_pts, step))
    if kept[-1] != n_pts - 1:
        kept.append(n_pts - 1)

    while len(kept) < min_pts:
        max_gap = 0
        max_idx = -1
        for i in range(len(kept) - 1):
            gap = kept[i + 1] - kept[i]
            if gap > max_gap:
                max_gap = gap
                max_idx = i
        if max_gap <= 1:
            break
        mid = (kept[max_idx] + kept[max_idx + 1]) // 2
        kept.insert(max_idx + 1, mid)
    return kept


def subsample_polydata(
    source: vtk.vtkPolyData, level: int, min_pts: int = 3
) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    """For each polyline cell of `source`, keep every (2**level)-th point.

    Returns:
      new_polydata: vtkPolyData with just the kept points and one polyline
        cell per source cell (when >=2 points remain). All source point-data
        arrays are subsetted to the kept points.
      kept_orig: 1D int array of original point IDs in the order matching the
        new point indexing (new_index -> original_id).
      sequential_pairs: (M, 2) int pairs (a, b) with a < b in NEW indexing,
        for consecutive points within each cell's kept subset.
    """
    if level < 0:
        raise ValueError(f"level must be >= 0 (got {level})")
    step = 1 << level

    new_points = vtk.vtkPoints()
    new_lines = vtk.vtkCellArray()
    kept_orig: list[int] = []
    seq_local: list[tuple[int, int]] = []
    cell_tract_labels: list[int] = []
    src_points = source.GetPoints()
    src_fiber_label = source.GetPointData().GetArray("FiberLabel")
    src_labels_np = (
        vtk_to_numpy(src_fiber_label) if src_fiber_label is not None else None
    )

    for c in range(source.GetNumberOfCells()):
        cell = source.GetCell(c)
        n_pts = cell.GetNumberOfPoints()
        ids = [cell.GetPointId(k) for k in range(n_pts)]
        local_keep = select_subsample_indices(n_pts, step, min_pts=min_pts)
        sub_ids = [ids[k] for k in local_keep]
        if not sub_ids:
            continue

        offset = len(kept_orig)
        for oid in sub_ids:
            new_points.InsertNextPoint(src_points.GetPoint(oid))
            kept_orig.append(oid)

        if len(sub_ids) >= 2:
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(sub_ids))
            for i in range(len(sub_ids)):
                polyline.GetPointIds().SetId(i, offset + i)
            new_lines.InsertNextCell(polyline)
            for i in range(1, len(sub_ids)):
                seq_local.append((offset + i - 1, offset + i))
            cell_tract_labels.append(
                int(src_labels_np[sub_ids[0]]) if src_labels_np is not None else c
            )

    kept_orig_arr = np.asarray(kept_orig, dtype=np.int64)
    seq = (
        np.asarray(seq_local, dtype=np.int64)
        if seq_local
        else np.zeros((0, 2), dtype=np.int64)
    )

    new_pd = vtk.vtkPolyData()
    new_pd.SetPoints(new_points)
    new_pd.SetLines(new_lines)

    if cell_tract_labels:
        cell_label_arr = numpy_to_vtk(
            np.asarray(cell_tract_labels, dtype=np.int32),
            deep=True,
            array_type=vtk.VTK_INT,
        )
        cell_label_arr.SetName("FiberLabel")
        new_pd.GetCellData().AddArray(cell_label_arr)
        new_pd.GetCellData().SetActiveScalars("FiberLabel")

    src_pdata = source.GetPointData()
    new_pdata = new_pd.GetPointData()
    for i in range(src_pdata.GetNumberOfArrays()):
        src_arr = src_pdata.GetArray(i)
        if src_arr is None:
            continue
        name = src_arr.GetName()
        if not name:
            continue
        np_arr = vtk_to_numpy(src_arr)
        if np_arr.ndim == 1:
            sub_np = np_arr[kept_orig_arr]
        else:
            sub_np = np_arr[kept_orig_arr, :]
        new_vtk_arr = numpy_to_vtk(
            sub_np, deep=True, array_type=src_arr.GetDataType()
        )
        new_vtk_arr.SetName(name)
        new_pdata.AddArray(new_vtk_arr)

    return new_pd, kept_orig_arr, seq


def write_polydata(
    polydata: vtk.vtkPolyData, path: Path, binary: bool, legacy: bool
) -> None:
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    if legacy:
        writer.SetFileVersion(vtk.vtkPolyDataWriter.VTK_LEGACY_READER_VERSION_4_2)
    writer.Write()


def build_adjacency(
    axis_points: np.ndarray,
    axis_labels: np.ndarray,
    sequential_pairs: np.ndarray,
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

    seq_set = (
        set(map(tuple, sequential_pairs)) if sequential_pairs.size else set()
    )
    extra_seq = seq_set - selected
    print(
        f"  {sequential_pairs.shape[0]:,} sequential edges along polylines "
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
        "sequential_edges": int(sequential_pairs.shape[0]),
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
            "Output dense adjacency text matrix path (only used when --levels=1). "
            "Defaults to the .npz path with a .txt extension. Each row is "
            "space-separated 0/1 values written via numpy.savetxt."
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=1,
        help=(
            "Number of resolution levels to generate (default: 1 = single "
            "graph, no suffix). For levels > 1, level L keeps every (2**L)-th "
            "point along each axis polyline, and each level's outputs get a "
            "_L<n> suffix."
        ),
    )
    parser.add_argument(
        "--min-points-per-fiber",
        type=int,
        default=3,
        help=(
            "Minimum number of points to keep on each fiber axis at every "
            "resolution level (default: 3). The first and last point of each "
            "fiber are always retained; midpoints are added as needed."
        ),
    )
    parser.add_argument(
        "--no-vtk",
        dest="write_vtk",
        action="store_false",
        help="Skip writing the per-level fiber-axis VTK file.",
    )
    parser.set_defaults(write_vtk=True)
    parser.add_argument(
        "--vtk-binary",
        action="store_true",
        help="Write per-level VTK in binary format (default: ASCII).",
    )
    parser.add_argument(
        "--vtk-legacy",
        action="store_true",
        help=(
            "Write per-level VTK using legacy file format version 4.2 "
            "instead of the default 5.1."
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

    if args.levels < 1:
        raise SystemExit(f"--levels must be >= 1 (got {args.levels})")

    nearest_full = cross_tract_nearest_distances(
        axis_pts,
        axis_lbl,
        axis_arc,
        fb_pts,
        fb_lbl,
        fb_arc,
        arc_tolerance=args.arc_tolerance,
        k_start=args.k_neighbors,
    )
    print()

    for level in range(args.levels):
        suffix = f"_L{level}" if args.levels > 1 else ""
        level_npz = out_npz.with_name(f"{out_npz.stem}{suffix}{out_npz.suffix}")
        level_info = level_npz.with_suffix(".info.txt")
        level_txt = (
            args.text_output
            if args.levels == 1 and args.text_output is not None
            else level_npz.with_suffix(".txt")
        )
        level_indices = level_npz.with_suffix(".indices.txt")

        print("=" * 64)
        print(f"Level {level} (step = {1 << level})")
        print("=" * 64)

        sub_polydata, kept_orig, seq_pairs_new = subsample_polydata(
            axis_pd, level, min_pts=args.min_points_per_fiber
        )
        sub_pts = axis_pts[kept_orig]
        sub_lbl = axis_lbl[kept_orig]
        sub_arc = axis_arc[kept_orig]
        sub_nearest = nearest_full[kept_orig]
        finite = sub_nearest[np.isfinite(sub_nearest)]
        n_failed = int(np.isinf(sub_nearest).sum())
        if not finite.size:
            print(f"Level {level}: no finite cross-tract distances; skipping.")
            continue
        median_nd = float(np.median(finite))
        threshold = args.threshold_multiplier * median_nd
        print(
            f"  axis points: {sub_pts.shape[0]:,} "
            f"(from {axis_pts.shape[0]:,} at L0)"
        )
        print(
            f"  nearest cross-tract: median={median_nd:.3f}, "
            f"p10={np.quantile(finite, 0.1):.3f}, "
            f"p90={np.quantile(finite, 0.9):.3f}  ({n_failed} missing)"
        )
        print(
            f"  threshold = {args.threshold_multiplier} × median "
            f"= {threshold:.3f} mm"
        )

        adj, stats = build_adjacency(sub_pts, sub_lbl, seq_pairs_new, threshold)
        avg_deg = adj.nnz / adj.shape[0] if adj.shape[0] else 0
        print(
            f"  adjacency: {adj.shape[0]} nodes, "
            f"{adj.nnz:,} non-zeros, "
            f"{stats['unique_edges']:,} unique undirected edges, "
            f"average degree {avg_deg:.1f}"
        )

        sp.save_npz(level_npz, adj)
        np.savetxt(level_txt, adj.toarray().astype(np.int8), fmt="%d")
        np.savetxt(level_indices, kept_orig, fmt="%d")

        if args.write_vtk:
            degrees = np.asarray(adj.sum(axis=1)).flatten().astype(np.int32)
            deg_arr = numpy_to_vtk(
                degrees, deep=True, array_type=vtk.VTK_INT
            )
            deg_arr.SetName("NumNeighbors")
            sub_polydata.GetPointData().AddArray(deg_arr)
            sub_polydata.GetPointData().SetActiveScalars("NumNeighbors")

            level_vtk = level_npz.with_suffix(".vtk")
            write_polydata(
                sub_polydata,
                level_vtk,
                binary=args.vtk_binary,
                legacy=args.vtk_legacy,
            )
        with level_info.open("w") as f:
            f.write(f"Input axis: {args.axis}\n")
            f.write(f"Input fibers: {args.fibers}\n")
            f.write(f"Resolution level: {level} (step = {1 << level})\n")
            f.write(f"Axis points at this level: {sub_pts.shape[0]}\n")
            f.write(f"Axis points at L0: {axis_pts.shape[0]}\n")
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
            f.write(
                f"Sequential edges along axes (candidates): "
                f"{stats['sequential_edges']}\n"
            )
            f.write(
                f"Sequential edges not already retained: "
                f"{stats['sequential_extra']}\n"
            )
            f.write(f"Unique undirected edges (union): {stats['unique_edges']}\n")
            f.write(f"Average degree: {avg_deg:.4f}\n")
        print(f"  saved: {level_npz}")
        print(f"         {level_txt}")
        print(f"         {level_indices}")
        print(f"         {level_info}")
        if args.write_vtk:
            print(f"         {level_vtk}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
