#!/usr/bin/env python3
"""Compute a 1D curved-line axis from a parametrized VTK fiber bundle.

Reads a VTK PolyData fiber file that contains a `SamplingDistance2Origin`
point-data array, bins points by arclength, averages the (x, y, z) coordinates
within each bin, and writes the resulting averaged points as a single polyline
ordered by arclength. The bin's arclength is stored as a `SamplingDistance2Origin`
point-data array on the output.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ARRAY_NAME = "SamplingDistance2Origin"


def find_arclength_array(polydata: vtk.vtkPolyData) -> vtk.vtkDataArray:
    point_data = polydata.GetPointData()
    array = point_data.GetArray(ARRAY_NAME)
    if array is None:
        array = point_data.GetAbstractArray(ARRAY_NAME)
    if array is None:
        available = [
            point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())
        ]
        raise SystemExit(
            f"Point-data array '{ARRAY_NAME}' not found. Available arrays: {available}"
        )
    return array


def compute_axis(
    points: np.ndarray,
    arclength: np.ndarray,
    bin_width: float,
    method: str,
    outlier_sigma: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if bin_width <= 0:
        raise SystemExit(f"--bin-width must be > 0 (got {bin_width})")
    if method not in {"mean", "median"}:
        raise SystemExit(f"Unknown --method: {method!r}")

    bin_indices = np.round(arclength / bin_width).astype(np.int64)
    unique_bins = np.unique(bin_indices)

    axis_points = np.zeros((unique_bins.shape[0], 3), dtype=np.float64)
    n_outliers = 0

    for i, bin_id in enumerate(unique_bins):
        bin_points = points[bin_indices == bin_id]
        median = np.median(bin_points, axis=0)

        if method == "median":
            axis_points[i] = median
            continue

        # Cleaned mean: drop points whose deviation from the median exceeds
        # outlier_sigma * std on any axis, then average the rest.
        std = bin_points.std(axis=0)
        deviations = np.abs(bin_points - median)
        outlier_axis = (std > 0) & (deviations > outlier_sigma * std)
        outlier_mask = outlier_axis.any(axis=1)
        inliers = bin_points[~outlier_mask]
        n_outliers += int(outlier_mask.sum())
        axis_points[i] = inliers.mean(axis=0) if inliers.size else median

    bin_centers = unique_bins.astype(np.float64) * bin_width
    order = np.argsort(bin_centers)
    return axis_points[order], bin_centers[order], n_outliers


def clip_endpoints(
    points: np.ndarray, arclength: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Clip points from either end whose adjacent step magnitudes exceed
    ``factor`` times the median step magnitude.

    Strategy: keep the longest contiguous run of "good" steps (those at most
    ``factor`` × median) and discard the points lying outside that run. This
    correctly handles "comet trail" cases where one or more endpoints cluster
    together off the main curve (so the very last step is small, but a step
    nearby is anomalously large).

    The median is recomputed after each peel so that once outlier steps are
    removed, a tighter threshold can catch any remaining mild anomalies.
    """
    if factor <= 0:
        raise SystemExit(f"--clip-factor must be > 0 (got {factor})")

    n_start_total = 0
    n_end_total = 0
    while points.shape[0] >= 3:
        step_mags = np.linalg.norm(np.diff(points, axis=0), axis=1)
        median = float(np.median(step_mags))
        if median <= 0:
            break
        threshold = factor * median
        good = step_mags <= threshold
        if good.all():
            break

        n = good.size
        best_start = 0
        best_len = 0
        i = 0
        while i < n:
            if not good[i]:
                i += 1
                continue
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) > best_len:
                best_len = j - i
                best_start = i
            i = j

        if best_len == 0:
            break

        lo = best_start
        hi = best_start + best_len + 1
        if lo == 0 and hi == points.shape[0]:
            break

        n_start_total += lo
        n_end_total += points.shape[0] - hi
        points = points[lo:hi]
        arclength = arclength[lo:hi]
    return points, arclength, n_start_total, n_end_total


def build_axis_polydata(
    mean_points: np.ndarray, arclength: np.ndarray
) -> vtk.vtkPolyData:
    n = mean_points.shape[0]

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(mean_points.astype(np.float32), deep=True))

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n)
    for i in range(n):
        lines.InsertCellPoint(i)

    arclength_array = numpy_to_vtk(arclength.astype(np.float32), deep=True)
    arclength_array.SetName(ARRAY_NAME)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(arclength_array)
    polydata.GetPointData().SetActiveScalars(ARRAY_NAME)
    return polydata


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read polydata or no points found in: {path}")
    return polydata


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a 1D axis curve from a parametrized VTK fiber bundle by "
            "averaging coordinates within arclength bins."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input VTK fiber file (PolyData with a SamplingDistance2Origin array).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output VTK file. Defaults to <input-stem>_axis.vtk in the input's directory.",
    )
    parser.add_argument(
        "-b",
        "--bin-width",
        type=float,
        default=1.0,
        help="Arclength bin width used for grouping points (default: 1.0).",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=("mean", "median"),
        default="mean",
        help=(
            "Per-bin aggregation method. 'mean' (default) drops outlier points "
            "whose deviation from the per-bin median exceeds --outlier-sigma "
            "standard deviations on any axis, then averages the rest. "
            "'median' uses the per-bin median coordinate directly."
        ),
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=3.0,
        help=(
            "Sigma threshold for outlier rejection in mean mode (default: 3.0). "
            "Ignored when --method median."
        ),
    )
    parser.add_argument(
        "--no-clip",
        dest="clip",
        action="store_false",
        help=(
            "Disable endpoint clipping. By default the axis is clipped: points "
            "are peeled from either end whenever the step to the next inward "
            "point exceeds --clip-factor times the median step magnitude."
        ),
    )
    parser.set_defaults(clip=True)
    parser.add_argument(
        "--clip-factor",
        type=float,
        default=4.0,
        help=(
            "Multiplier on the median step magnitude used as the clipping "
            "threshold (default: 4.0). Ignored when --no-clip is given."
        ),
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write the output VTK file in binary format (default: ASCII).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Write using VTK legacy file format version 4.2 (single-line LINES "
            "cells) instead of the default version 5.1 (OFFSETS/CONNECTIVITY)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_axis.vtk"
    )

    polydata = read_polydata(args.input)
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    arclength = vtk_to_numpy(find_arclength_array(polydata)).astype(np.float64)

    if points.shape[0] != arclength.shape[0]:
        raise SystemExit(
            f"Point count ({points.shape[0]}) does not match "
            f"{ARRAY_NAME} length ({arclength.shape[0]})."
        )

    axis_coords, axis_arclength, n_outliers = compute_axis(
        points, arclength, args.bin_width, args.method, args.outlier_sigma
    )

    n_clipped_start = n_clipped_end = 0
    if args.clip:
        axis_coords, axis_arclength, n_clipped_start, n_clipped_end = clip_endpoints(
            axis_coords, axis_arclength, args.clip_factor
        )

    axis = build_axis_polydata(axis_coords, axis_arclength)
    write_polydata(axis, output_path, binary=args.binary, legacy=args.legacy)

    suffix = (
        f", dropped {n_outliers} outliers (>{args.outlier_sigma}σ)"
        if args.method == "mean"
        else ""
    )
    if args.clip:
        suffix += (
            f", clipped {n_clipped_start} from start and {n_clipped_end} "
            f"from end (>{args.clip_factor}×median step)"
        )
    print(
        f"Wrote axis with {axis_coords.shape[0]} points "
        f"(bin width {args.bin_width}, method {args.method}{suffix}) "
        f"to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
