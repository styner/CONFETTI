#!/usr/bin/env python3
"""Evaluate per-subject imputation methods by holding out random tract sets
from subjects with complete data.

For each complete subject (no -1 entries in any property), this script:
  1. Repeatedly samples a random subset of fiber tracts (FiberLabel groups)
  2. Marks all that subject's points in those tracts as "held out"
  3. Fits each imputation method on the remaining (observed) points
  4. Predicts at the held-out points and records per-property MAE and RMSE
  5. Aggregates across trials and subjects and reports per-property comparisons

Methods compared:
  mean    -- subject's per-property mean of observed values
  median  -- subject's per-property median of observed values
  linear  -- 3D linear regression on (x, y, z)
  knn5    -- k-Nearest Neighbors regression (k=5, distance-weighted)
  knn10   -- k-Nearest Neighbors regression (k=10, distance-weighted)
  rbf     -- scipy RBFInterpolator (thin-plate spline, 50 neighbors per query)
  siren   -- per-subject SIREN INR matching impute_missing_profiles.py
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import vtk
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).parent))
from impute_missing_profiles import (  # noqa: E402
    collect_subjects,
    compute_property_stats,
    detect_properties,
    normalize_coords,
    select_device,
    train_siren,
)


# ---------------------------------------------------------------------------
# Imputation methods. All take (coords_obs, vals_obs, coords_miss) in
# normalized-coords / z-scored-values space and return predictions in that
# same space, with shape (n_miss, n_props).
# ---------------------------------------------------------------------------


def impute_mean(coords_obs, vals_obs, coords_miss):
    mean = vals_obs.mean(axis=0, keepdims=True)
    return np.broadcast_to(mean, (coords_miss.shape[0], vals_obs.shape[1])).copy()


def impute_median(coords_obs, vals_obs, coords_miss):
    med = np.median(vals_obs, axis=0, keepdims=True)
    return np.broadcast_to(med, (coords_miss.shape[0], vals_obs.shape[1])).copy()


def impute_linear(coords_obs, vals_obs, coords_miss):
    reg = LinearRegression()
    reg.fit(coords_obs, vals_obs)
    return reg.predict(coords_miss)


def impute_knn(coords_obs, vals_obs, coords_miss, k):
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(coords_obs, vals_obs)
    return knn.predict(coords_miss)


def impute_rbf(coords_obs, vals_obs, coords_miss):
    rbf = RBFInterpolator(
        coords_obs, vals_obs, kernel="thin_plate_spline", neighbors=50
    )
    return rbf(coords_miss)


def impute_siren(
    coords_obs, vals_obs, coords_miss, *, device, siren_kwargs, seed
):
    coords_obs_t = torch.tensor(coords_obs, dtype=torch.float32, device=device)
    vals_obs_t = torch.tensor(vals_obs, dtype=torch.float32, device=device)
    coords_miss_t = torch.tensor(coords_miss, dtype=torch.float32, device=device)
    model, _ = train_siren(
        coords_obs_t,
        vals_obs_t,
        n_outputs=vals_obs.shape[1],
        device=device,
        seed=seed,
        **siren_kwargs,
    )
    model.eval()
    with torch.no_grad():
        return model(coords_miss_t).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


def find_complete_subjects(
    pdata: vtk.vtkPointData,
    subject_arrays: dict[str, dict[str, str]],
    missing_value: float,
) -> list[str]:
    complete = []
    for subj, prop_to_name in subject_arrays.items():
        is_complete = True
        for name in prop_to_name.values():
            arr = vtk_to_numpy(pdata.GetArray(name))
            if (arr == missing_value).any():
                is_complete = False
                break
        if is_complete:
            complete.append(subj)
    return sorted(complete)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate imputation methods by holding out random tracts from "
            "subjects with complete data and measuring reconstruction error."
        )
    )
    parser.add_argument("input", type=Path, help="Input merged VTK file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Per-trial detail CSV. Defaults to <input-stem>_eval.csv next to "
            "the input."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Aggregated summary CSV (mean MAE/RMSE per method-property).",
    )
    parser.add_argument(
        "--missing-value", type=float, default=-1.0,
        help="Sentinel marking missing entries in the input (default: -1).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Random holdout trials per subject (default: 5).",
    )
    parser.add_argument(
        "--holdout-tracts",
        type=int,
        default=10,
        help="Number of tracts to hold out per trial (default: 10).",
    )
    parser.add_argument(
        "--label-array",
        default="FiberLabel",
        help="Name of the per-point tract-label array (default: FiberLabel).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N complete subjects (smoke-test).",
    )
    parser.add_argument(
        "--methods",
        default="mean,median,linear,knn5,knn10,rbf,siren",
        help="Comma-separated subset of methods to run.",
    )
    parser.add_argument(
        "--siren-epochs",
        type=int,
        default=500,
        help="SIREN epochs per trial (default: 500; lower than impute script for eval speed).",
    )
    parser.add_argument("--siren-hidden-features", type=int, default=256)
    parser.add_argument("--siren-hidden-layers", type=int, default=3)
    parser.add_argument("--siren-omega-0", type=float, default=30.0)
    parser.add_argument("--siren-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def build_methods(args: argparse.Namespace, device: torch.device):
    siren_kwargs = dict(
        hidden_features=args.siren_hidden_features,
        hidden_layers=args.siren_hidden_layers,
        omega_0=args.siren_omega_0,
        epochs=args.siren_epochs,
        lr=args.siren_lr,
    )
    catalog = {
        "mean": lambda co, vo, cm, seed: impute_mean(co, vo, cm),
        "median": lambda co, vo, cm, seed: impute_median(co, vo, cm),
        "linear": lambda co, vo, cm, seed: impute_linear(co, vo, cm),
        "knn5": lambda co, vo, cm, seed: impute_knn(co, vo, cm, k=5),
        "knn10": lambda co, vo, cm, seed: impute_knn(co, vo, cm, k=10),
        "rbf": lambda co, vo, cm, seed: impute_rbf(co, vo, cm),
        "siren": lambda co, vo, cm, seed: impute_siren(
            co, vo, cm, device=device, siren_kwargs=siren_kwargs, seed=seed
        ),
    }
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in requested if m not in catalog]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {list(catalog)}")
    return {m: catalog[m] for m in requested}


def print_summary(
    rows: list[dict],
    properties: list[str],
    methods: list[str],
    prop_stats: dict[str, tuple[float, float]],
) -> None:
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["method"], r["property"])].append((r["mae"], r["rmse"]))

    name_w = max(len(m) for m in methods)
    print()
    print("=" * 78)
    print("Per-property method comparison (lower is better)")
    print("=" * 78)
    for prop in properties:
        prop_std = prop_stats[prop][1]
        print(f"\n{prop}  (property std over all observed = {prop_std:.4g})")
        print(f"  {'method':<{name_w}}   {'MAE':>10}   {'NMAE':>8}   {'RMSE':>10}   {'n_eval'}")
        print(f"  {'-'*name_w}   {'-'*10}   {'-'*8}   {'-'*10}   {'-'*6}")
        rankings = []
        for method in methods:
            vals = by_key.get((method, prop), [])
            if not vals:
                continue
            mae = np.mean([v[0] for v in vals])
            rmse = np.mean([v[1] for v in vals])
            rankings.append((method, mae, rmse, len(vals)))
        rankings.sort(key=lambda r: r[1])
        for method, mae, rmse, n in rankings:
            nmae = mae / prop_std if prop_std > 0 else float("nan")
            print(f"  {method:<{name_w}}   {mae:>10.4g}   {nmae:>8.3f}   {rmse:>10.4g}   {n:>6}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(args.input))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read polydata: {args.input}")
    pdata = polydata.GetPointData()

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    coords_norm = normalize_coords(points)
    n_points = points.shape[0]

    labels_arr = pdata.GetArray(args.label_array)
    if labels_arr is None:
        raise SystemExit(
            f"Label array '{args.label_array}' not found on input. "
            f"Available: {[pdata.GetArrayName(i) for i in range(pdata.GetNumberOfArrays())][:5]}..."
        )
    labels = vtk_to_numpy(labels_arr)
    unique_labels = np.unique(labels)
    if args.holdout_tracts >= len(unique_labels):
        raise SystemExit(
            f"--holdout-tracts ({args.holdout_tracts}) must be < number of "
            f"tracts ({len(unique_labels)})."
        )

    array_names = [pdata.GetArrayName(i) for i in range(pdata.GetNumberOfArrays())]
    properties = detect_properties(array_names)
    if not properties:
        raise SystemExit("No property arrays detected.")
    subject_arrays = collect_subjects(array_names, properties)
    prop_stats = compute_property_stats(pdata, subject_arrays, properties, args.missing_value)

    complete_subjects = find_complete_subjects(pdata, subject_arrays, args.missing_value)
    print(f"Found {len(complete_subjects)} complete subjects (no missing data)")
    if args.limit is not None:
        complete_subjects = complete_subjects[: args.limit]
        print(f"Limiting to first {len(complete_subjects)} via --limit")
    if not complete_subjects:
        raise SystemExit("No complete subjects available for evaluation.")

    methods = build_methods(args, device)
    print(f"Evaluating methods: {list(methods)}")
    print(
        f"Holding out {args.holdout_tracts}/{len(unique_labels)} tracts "
        f"per trial; {args.n_trials} trials per subject; "
        f"{len(complete_subjects)} subjects -> "
        f"{len(complete_subjects) * args.n_trials} total trials"
    )

    rows: list[dict] = []
    timing: dict[str, float] = defaultdict(float)
    total_start = time.time()

    for s_idx, subj in enumerate(complete_subjects, start=1):
        prop_to_name = subject_arrays[subj]
        prop_keys = sorted(prop_to_name.keys())
        n_props = len(prop_keys)

        vals_norm = np.zeros((n_points, n_props), dtype=np.float32)
        for j, prop in enumerate(prop_keys):
            arr = vtk_to_numpy(pdata.GetArray(prop_to_name[prop])).astype(np.float64)
            mu, sd = prop_stats[prop]
            vals_norm[:, j] = (arr - mu) / sd

        for trial in range(args.n_trials):
            held = rng.choice(
                unique_labels, size=args.holdout_tracts, replace=False
            )
            mask_miss = np.isin(labels, held)
            mask_obs = ~mask_miss
            if mask_obs.sum() < 10 or mask_miss.sum() == 0:
                continue

            co = coords_norm[mask_obs]
            vo = vals_norm[mask_obs]
            cm = coords_norm[mask_miss]
            vt = vals_norm[mask_miss]

            for method_name, method_fn in methods.items():
                t0 = time.time()
                try:
                    preds_norm = method_fn(
                        co, vo, cm, args.seed + s_idx * 1000 + trial
                    )
                except Exception as exc:  # pragma: no cover - logged not crashed
                    print(f"  [{subj} t{trial} {method_name}] failed: {exc}")
                    continue
                timing[method_name] += time.time() - t0

                for j, prop in enumerate(prop_keys):
                    mu, sd = prop_stats[prop]
                    true_real = vt[:, j] * sd + mu
                    pred_real = preds_norm[:, j] * sd + mu
                    mae = float(np.mean(np.abs(pred_real - true_real)))
                    rmse = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))
                    rows.append(
                        {
                            "subject": subj,
                            "trial": trial,
                            "method": method_name,
                            "property": prop,
                            "n_points": int(mask_miss.sum()),
                            "mae": mae,
                            "rmse": rmse,
                        }
                    )

        if s_idx % 5 == 0 or s_idx == len(complete_subjects):
            elapsed = time.time() - total_start
            rate = s_idx / elapsed if elapsed > 0 else 0
            eta = (len(complete_subjects) - s_idx) / rate if rate > 0 else 0
            print(
                f"[{s_idx}/{len(complete_subjects)}] {rate:.2f} subj/s "
                f"elapsed {elapsed:.0f}s ETA {eta:.0f}s"
            )

    print()
    print("Per-method total wall time (s):")
    for m, t in sorted(timing.items(), key=lambda x: -x[1]):
        print(f"  {m:<10}  {t:.1f}")

    output_csv = args.output or args.input.with_name(f"{args.input.stem}_eval.csv")
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject", "trial", "method", "property",
                "n_points", "mae", "rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetail CSV: {output_csv}")

    if args.summary or True:
        summary_path = args.summary or args.input.with_name(
            f"{args.input.stem}_eval_summary.csv"
        )
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["method"], r["property"])].append((r["mae"], r["rmse"]))
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "method", "property", "n_eval",
                "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                "property_std", "nmae",
            ])
            for (method, prop), vals in sorted(by_key.items()):
                mae_arr = np.array([v[0] for v in vals])
                rmse_arr = np.array([v[1] for v in vals])
                prop_std = prop_stats[prop][1]
                w.writerow([
                    method, prop, len(vals),
                    f"{mae_arr.mean():.6g}", f"{mae_arr.std():.6g}",
                    f"{rmse_arr.mean():.6g}", f"{rmse_arr.std():.6g}",
                    f"{prop_std:.6g}",
                    f"{(mae_arr.mean() / prop_std):.6g}" if prop_std else "nan",
                ])
        print(f"Summary CSV: {summary_path}")

    print_summary(rows, properties, list(methods), prop_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
