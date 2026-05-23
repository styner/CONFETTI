#!/usr/bin/env python3
"""Impute missing property values in a merged fiber-axis VTK using per-subject
SIREN implicit neural representations.

Reads a VTK PolyData file (output of merge_fibers.py) whose point-data arrays
follow the convention `<property>_<subject>`. For each subject with sentinel
(default -1) entries, trains a per-subject SIREN with 3D position (x, y, z) as
input and the subject's available properties (typically 7) as the output head,
then predicts at missing positions, denormalizes, and writes a new VTK file.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = (6.0 / in_features) ** 0.5 / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)
        ]
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0
                )
            )
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = (6.0 / hidden_features) ** 0.5 / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def detect_properties(names: list[str]) -> list[str]:
    """Property prefixes are first underscore-delimited tokens that appear with
    multiple distinct subject suffixes."""
    by_prefix: dict[str, set[str]] = defaultdict(set)
    for n in names:
        if "_" not in n:
            continue
        prefix, rest = n.split("_", 1)
        by_prefix[prefix].add(rest)
    return sorted(p for p, subs in by_prefix.items() if len(subs) >= 2)


def collect_subjects(
    names: list[str], properties: list[str]
) -> dict[str, dict[str, str]]:
    """Returns subject -> {property: array_name}."""
    prop_set = set(properties)
    subject_arrays: dict[str, dict[str, str]] = defaultdict(dict)
    for n in names:
        if "_" not in n:
            continue
        prefix, rest = n.split("_", 1)
        if prefix not in prop_set:
            continue
        subject_arrays[rest][prefix] = n
    return subject_arrays


def select_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_coords(points: np.ndarray) -> np.ndarray:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = (bbox_max - bbox_min) / 2.0
    scale = np.where(scale > 0, scale, 1.0)
    return ((points - center) / scale).astype(np.float32)


def compute_property_stats(
    pdata: vtk.vtkPointData,
    subject_arrays: dict[str, dict[str, str]],
    properties: list[str],
    missing_value: float,
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for prop in properties:
        observed_chunks: list[np.ndarray] = []
        for prop_to_name in subject_arrays.values():
            name = prop_to_name.get(prop)
            if name is None:
                continue
            arr = vtk_to_numpy(pdata.GetArray(name))
            observed_chunks.append(arr[arr != missing_value])
        if not observed_chunks:
            stats[prop] = (0.0, 1.0)
            continue
        all_obs = np.concatenate(observed_chunks)
        mu = float(all_obs.mean())
        sd = float(all_obs.std())
        if sd == 0.0:
            sd = 1.0
        stats[prop] = (mu, sd)
    return stats


def train_siren(
    coords_obs: torch.Tensor,
    vals_obs: torch.Tensor,
    n_outputs: int,
    *,
    hidden_features: int,
    hidden_layers: int,
    omega_0: float,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[SIREN, float]:
    torch.manual_seed(seed)
    model = SIREN(
        in_features=3,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=n_outputs,
        first_omega_0=omega_0,
        hidden_omega_0=omega_0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    final_loss = float("nan")
    for _ in range(epochs):
        pred = model(coords_obs)
        loss = ((pred - vals_obs) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return model, final_loss


def replace_array(
    pdata: vtk.vtkPointData, name: str, data: np.ndarray
) -> None:
    new_arr = numpy_to_vtk(data.astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT)
    new_arr.SetName(name)
    pdata.RemoveArray(name)
    pdata.AddArray(new_arr)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Impute missing per-subject property values in a merged fiber-axis "
            "VTK using per-subject SIREN implicit neural representations."
        )
    )
    parser.add_argument(
        "input", type=Path, help="Input merged VTK file (e.g. FiberAxisProfiles_merged.vtk)."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output VTK file. Defaults to <input-stem>_imputed.vtk next to the input.",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=-1.0,
        help="Sentinel value marking missing entries (default: -1).",
    )
    parser.add_argument(
        "--hidden-features",
        type=int,
        default=256,
        help="SIREN hidden layer width (default: 256).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=3,
        help="Number of SIREN hidden sine layers (default: 3).",
    )
    parser.add_argument(
        "--omega-0",
        type=float,
        default=30.0,
        help="SIREN omega_0 (frequency scale) for all sine layers (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Training epochs per subject (default: 1500).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Torch device (default: auto -> cuda > mps > cpu).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only impute the first N subjects (smoke-test).",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write output in VTK binary format (default: ASCII).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Write output in VTK legacy 4.2 format instead of 5.1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(args.input))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read or empty polydata: {args.input}")
    pdata = polydata.GetPointData()
    n_points = polydata.GetNumberOfPoints()
    print(f"Loaded {n_points} points, {pdata.GetNumberOfArrays()} point-data arrays")

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    coords_norm = normalize_coords(points)
    coords_t = torch.tensor(coords_norm, dtype=torch.float32, device=device)

    array_names = [pdata.GetArrayName(i) for i in range(pdata.GetNumberOfArrays())]
    properties = detect_properties(array_names)
    if not properties:
        raise SystemExit("No property arrays detected (expected names like 'FA_<subject>').")
    print(f"Detected {len(properties)} properties: {properties}")

    subject_arrays = collect_subjects(array_names, properties)
    print(f"Detected {len(subject_arrays)} subjects")

    prop_stats = compute_property_stats(
        pdata, subject_arrays, properties, args.missing_value
    )
    for prop in properties:
        mu, sd = prop_stats[prop]
        print(f"  {prop}: mean={mu:.4g}, std={sd:.4g}")

    needs_imputation: list[str] = []
    n_already_complete = 0
    n_skipped_no_obs = 0
    for subj in sorted(subject_arrays.keys()):
        any_missing = False
        any_observed = False
        for name in subject_arrays[subj].values():
            arr = vtk_to_numpy(pdata.GetArray(name))
            if not any_missing and (arr == args.missing_value).any():
                any_missing = True
            if not any_observed and (arr != args.missing_value).any():
                any_observed = True
            if any_missing and any_observed:
                break
        if not any_missing:
            n_already_complete += 1
        elif not any_observed:
            n_skipped_no_obs += 1
        else:
            needs_imputation.append(subj)

    print(
        f"Subjects: {n_already_complete} complete (no imputation needed), "
        f"{n_skipped_no_obs} fully missing (cannot impute), "
        f"{len(needs_imputation)} need imputation"
    )

    if args.limit is not None:
        needs_imputation = needs_imputation[: args.limit]
        print(f"Limiting run to first {len(needs_imputation)} subjects via --limit")

    n_imputed = 0
    losses: list[float] = []
    start_time = time.time()

    for idx, subj in enumerate(needs_imputation, start=1):
        prop_to_name = subject_arrays[subj]
        prop_keys = sorted(prop_to_name.keys())
        n_out = len(prop_keys)

        vals = np.zeros((n_points, n_out), dtype=np.float32)
        mask_obs_per_prop = np.zeros((n_points, n_out), dtype=bool)
        for j, prop in enumerate(prop_keys):
            arr = vtk_to_numpy(pdata.GetArray(prop_to_name[prop]))
            mask = arr != args.missing_value
            mu, sd = prop_stats[prop]
            vals[:, j] = (arr - mu) / sd
            mask_obs_per_prop[:, j] = mask

        mask_obs = mask_obs_per_prop.all(axis=1)

        mask_obs_t = torch.from_numpy(mask_obs).to(device)
        coords_obs = coords_t[mask_obs_t]
        vals_obs = torch.tensor(vals[mask_obs], dtype=torch.float32, device=device)

        model, final_loss = train_siren(
            coords_obs,
            vals_obs,
            n_outputs=n_out,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            omega_0=args.omega_0,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            seed=args.seed + idx,
        )
        losses.append(final_loss)

        mask_miss = ~mask_obs
        coords_miss = coords_t[torch.from_numpy(mask_miss).to(device)]
        model.eval()
        with torch.no_grad():
            preds_norm = model(coords_miss).detach().cpu().numpy()

        for j, prop in enumerate(prop_keys):
            mu, sd = prop_stats[prop]
            arr = vtk_to_numpy(pdata.GetArray(prop_to_name[prop])).copy()
            arr[mask_miss] = preds_norm[:, j] * sd + mu
            replace_array(pdata, prop_to_name[prop], arr)

        n_imputed += 1
        if idx % 25 == 0 or idx == len(needs_imputation):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = len(needs_imputation) - idx
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"[{idx}/{len(needs_imputation)}] imputed={n_imputed} "
                f"last_train_mse={final_loss:.4f} "
                f"{rate:.2f} subj/s ETA {eta:.0f}s"
            )

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_imputed.vtk"
    )
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(polydata)
    if args.binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    if args.legacy:
        writer.SetFileVersion(vtk.vtkPolyDataWriter.VTK_LEGACY_READER_VERSION_4_2)
    writer.Write()

    print()
    print(
        f"Done. Imputed {n_imputed} subjects, "
        f"{n_already_complete} already complete, "
        f"{n_skipped_no_obs} skipped (no observed data)."
    )
    if losses:
        arr = np.array(losses)
        print(
            f"Per-subject final training MSE (z-scored): "
            f"median={np.median(arr):.4f}, "
            f"p90={np.quantile(arr, 0.9):.4f}, "
            f"max={arr.max():.4f}"
        )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
