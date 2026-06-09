"""Spatial interpretability for the GNN predictions.

Three complementary tools:

  permutation_importance()  -- per-node and per-tract, model-agnostic.
      Shuffles one node's (or one tract's) feature channels across the test
      set; measures drop in test AUC. The largest drops are the most
      important spatial regions for the classifier as a whole.

  integrated_gradients()    -- per-subject attribution.
      Standard IG: integrate the gradient of the model output w.r.t. the input
      features along a linear path from a baseline (zero) to the input.
      Returns (n_subjects, n_nodes, n_features) attribution tensor.

  gnn_explainer_sparse()    -- per-prediction sparse explanation.
      Wraps PyG's GNNExplainer to surface a minimal node/feature mask that
      preserves the model's prediction for a single subject. Useful for case
      studies and clinical narratives.

Both per-node permutation importance and integrated-gradients results can be
written back to a VTK as a new point-data array so the spatial pattern can be
inspected in ParaView (see `write_importance_vtk` in `data` module integrations
via classify_ds.py).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import vtk
from sklearn.metrics import roc_auc_score
from vtk.util.numpy_support import numpy_to_vtk

from .config import PROPERTIES
from .data import LevelData
from .models import HierarchicalGraphUNet, make_level_batch, make_pool_map_batch


# ---------------------------------------------------------------------------
# Permutation importance (per node and per tract)
# ---------------------------------------------------------------------------


def _predict(
    model: nn.Module,
    feats_by_level: dict[int, np.ndarray],
    levels: dict[int, LevelData],
    used_levels: list[int],
    pool_maps_list: list[np.ndarray],
    is_unet: bool,
    device: torch.device,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    B = next(iter(feats_by_level.values())).shape[0]
    batches = []
    for L in used_levels:
        bd = make_level_batch(
            feats_by_level[L], levels[L].edge_index, levels[L].edge_weight, device
        )
        batches.append(bd)
    cov_t = (
        torch.as_tensor(covariates, dtype=torch.float32, device=device)
        if covariates is not None
        else None
    )
    if is_unet:
        sorted_used = sorted(used_levels)
        pm_batched = [
            make_pool_map_batch(pm, levels[Lf].coords.shape[0], B, device)
            for pm, Lf in zip(pool_maps_list, sorted_used[:-1])
        ]
        with torch.no_grad():
            out = model(batches, pm_batched, covariates=cov_t)
    else:
        with torch.no_grad():
            out = model(batches, covariates=cov_t)
    return out.cpu().numpy()


def permutation_importance(
    model: nn.Module,
    feats_by_level: dict[int, np.ndarray],
    y_test: np.ndarray,
    levels: dict[int, LevelData],
    used_levels: list[int],
    pool_maps_list: list[np.ndarray],
    metric: str = "auc",
    n_repeats: int = 5,
    granularity: str = "tract",
    seed: int = 0,
    device: torch.device | None = None,
    properties: tuple[str, ...] = PROPERTIES,
    covariates: np.ndarray | None = None,
    covariate_names: tuple[str, ...] | None = None,
) -> dict:
    """Permutation importance.

    `granularity`:
      - "tract": shuffle property values of all nodes belonging to one tract
        across subjects (the tract id is taken from level0 fiber_label). Most
        useful for clinical narrative. Returns a dict[tract_id -> importance].
      - "node": shuffle property values of one node across subjects (returns
        a dict[node_index -> importance], using the first level in
        `used_levels` for indexing).

    The importance is the metric loss: importance = baseline_metric -
    metric_after_shuffle. Higher = more important.
    """
    device = device or torch.device("cpu")
    is_unet = isinstance(model, HierarchicalGraphUNet)
    rng = np.random.default_rng(seed)

    base_logits = _predict(
        model, feats_by_level, levels, used_levels, pool_maps_list, is_unet, device,
        covariates=covariates,
    )
    n_prop = len(properties)

    def score(logits: np.ndarray) -> float:
        if metric == "auc":
            probs = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))
            if len(np.unique(y_test)) < 2:
                return float("nan")
            return float(roc_auc_score(y_test, probs))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    base_metric = score(base_logits)

    # Covariate-level permutation: one entry per covariate column, shuffled
    # across subjects, while node features stay at their original values.
    if granularity == "covariate":
        if covariates is None:
            raise ValueError("granularity='covariate' requires covariates to be passed in")
        names = covariate_names or tuple(f"cov_{i}" for i in range(covariates.shape[1]))
        out_cov: dict[str, float] = {}
        for c_idx, name in enumerate(names):
            drops = []
            for _ in range(n_repeats):
                cov_shuf = covariates.copy()
                perm = rng.permutation(cov_shuf.shape[0])
                cov_shuf[:, c_idx] = covariates[perm, c_idx]
                logits = _predict(
                    model, feats_by_level, levels, used_levels, pool_maps_list,
                    is_unet, device, covariates=cov_shuf,
                )
                drops.append(base_metric - score(logits))
            with np.errstate(invalid="ignore"):
                out_cov[name] = float(np.nanmean(drops))
        return out_cov

    out: dict[int, float] = {}
    L_for_indexing = used_levels[0]
    fiber_label = levels[L_for_indexing].fiber_label
    if granularity == "tract":
        groups = {
            int(t): np.where(fiber_label == t)[0].tolist()
            for t in np.unique(fiber_label)
        }
    elif granularity == "node":
        groups = {int(i): [i] for i in range(fiber_label.shape[0])}
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    iter_count = 0
    for gid, node_idx in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = {L: feats_by_level[L].copy() for L in used_levels}
            # Only shuffle on the indexing level; coarser levels (if any) keep
            # their original spatial layout. This matches how an interpreter
            # would think about per-tract attribution.
            x = shuffled[L_for_indexing]
            S = x.shape[0]
            perm = rng.permutation(S)
            x[:, node_idx, :n_prop] = x[perm][:, node_idx, :n_prop]
            shuffled[L_for_indexing] = x
            try:
                logits = _predict(
                    model, shuffled, levels, used_levels, pool_maps_list, is_unet, device,
                    covariates=covariates,
                )
                drops.append(base_metric - score(logits))
            except RuntimeError as exc:
                print(f"  perm-imp draw failed at gid={gid}: {exc}")
                drops.append(float("nan"))
            iter_count += 1
            if iter_count % 64 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
        with np.errstate(invalid="ignore"):
            out[gid] = float(np.nanmean(drops))
    return out


# ---------------------------------------------------------------------------
# Integrated Gradients (per-subject feature attribution)
# ---------------------------------------------------------------------------


def integrated_gradients(
    model: nn.Module,
    feats_by_level: dict[int, np.ndarray],
    levels: dict[int, LevelData],
    used_levels: list[int],
    pool_maps_list: list[np.ndarray],
    target_idx: int = 0,
    steps: int = 32,
    device: torch.device | None = None,
    covariates: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Integrated Gradients on the input features of the first ("primary")
    level. Returns dict[L -> (S, N, F) attribution array].

    Baseline = zero tensor; the path is linear from baseline to input. For
    multi-output models, `target_idx` picks the output dimension to attribute.

    Note: gradients only flow through the level whose tensor is differentiable;
    here we attribute through ALL levels by making each level's input require
    grad and integrating each separately. Spatial features are included in the
    attribution (they're constants per-level, so per-channel contributions are
    informative even though their values don't differ per-subject).
    """
    device = device or torch.device("cpu")
    is_unet = isinstance(model, HierarchicalGraphUNet)
    model.eval()
    B = next(iter(feats_by_level.values())).shape[0]

    inputs = {
        L: torch.from_numpy(feats_by_level[L]).to(device).requires_grad_(True)
        for L in used_levels
    }
    baselines = {L: torch.zeros_like(inputs[L]) for L in used_levels}
    attrs = {L: torch.zeros_like(inputs[L]) for L in used_levels}

    for step in range(steps):
        alpha = (step + 0.5) / steps
        interp = {L: baselines[L] + alpha * (inputs[L] - baselines[L]) for L in used_levels}
        for L in used_levels:
            interp[L].requires_grad_(True)

        batches = []
        for L in used_levels:
            bd = make_level_batch(
                interp[L].detach().cpu().numpy(),
                levels[L].edge_index,
                levels[L].edge_weight,
                device,
            )
            # Replace the batched x with the live tensor (preserve grad).
            B_, N_, _ = interp[L].shape
            bd["x"] = interp[L].reshape(B_ * N_, -1)
            batches.append(bd)

        cov_t = (
            torch.as_tensor(covariates, dtype=torch.float32, device=device)
            if covariates is not None
            else None
        )
        if is_unet:
            sorted_used = sorted(used_levels)
            pm_batched = [
                make_pool_map_batch(pm, levels[Lf].coords.shape[0], B, device)
                for pm, Lf in zip(pool_maps_list, sorted_used[:-1])
            ]
            out = model(batches, pm_batched, covariates=cov_t)
        else:
            out = model(batches, covariates=cov_t)
        if out.dim() == 2:
            out = out[:, target_idx]
        out = out.sum()

        grads = torch.autograd.grad(out, list(interp.values()), allow_unused=True)
        for L, g in zip(interp.keys(), grads):
            if g is not None:
                attrs[L] = attrs[L] + g.detach() / steps

    out_np: dict[int, np.ndarray] = {}
    for L in used_levels:
        out_np[L] = (attrs[L] * (inputs[L] - baselines[L])).detach().cpu().numpy()
    return out_np


# ---------------------------------------------------------------------------
# GNNExplainer (per-prediction sparse subgraph)
# ---------------------------------------------------------------------------


def gnn_explainer_sparse(
    model: nn.Module,
    feats_by_level: dict[int, np.ndarray],
    subject_index: int,
    levels: dict[int, LevelData],
    used_levels: list[int],
    pool_maps_list: list[np.ndarray],
    epochs: int = 200,
    device: torch.device | None = None,
    covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GNNExplainer on a single subject, returning per-node and per-feature
    masks on the primary (finest) used level. Pure-Python fallback for cases
    where PyG's Explainer wrapper is awkward to plug into multi-level inputs.
    """
    device = device or torch.device("cpu")
    is_unet = isinstance(model, HierarchicalGraphUNet)

    # Restrict to the chosen subject (B=1).
    feats_one: dict[int, np.ndarray] = {
        L: feats_by_level[L][subject_index : subject_index + 1] for L in used_levels
    }
    L_primary = used_levels[0]
    n_nodes = feats_one[L_primary].shape[1]
    n_feat = feats_one[L_primary].shape[2]

    # Learnable masks (softmax-bounded via sigmoid).
    node_logits = torch.zeros(n_nodes, device=device, requires_grad=True)
    feat_logits = torch.zeros(n_feat, device=device, requires_grad=True)
    optim = torch.optim.Adam([node_logits, feat_logits], lr=0.1)

    def masked_features() -> dict[int, np.ndarray]:
        """Apply the learnable node mask at the primary (finest) level only --
        coarser levels have a different node count, so a per-node mask is not
        directly applicable there. The per-feature mask is shared across all
        levels because it indexes the same property/spatial channels.
        """
        nm = torch.sigmoid(node_logits)
        fm = torch.sigmoid(feat_logits)
        out: dict[int, torch.Tensor] = {}
        for L in used_levels:
            x = torch.as_tensor(feats_one[L], dtype=torch.float32, device=device)
            if L == L_primary:
                x = x * nm.unsqueeze(0).unsqueeze(-1) * fm.unsqueeze(0).unsqueeze(0)
            else:
                x = x * fm.unsqueeze(0).unsqueeze(0)
            out[L] = x
        return out

    cov_t = (
        torch.as_tensor(covariates, dtype=torch.float32, device=device)
        if covariates is not None
        else None
    )

    model.eval()
    with torch.no_grad():
        bd_full = []
        for L in used_levels:
            bd = make_level_batch(
                feats_one[L], levels[L].edge_index, levels[L].edge_weight, device
            )
            bd_full.append(bd)
        if is_unet:
            sorted_used = sorted(used_levels)
            pm_batched = [
                make_pool_map_batch(pm, levels[Lf].coords.shape[0], 1, device)
                for pm, Lf in zip(pool_maps_list, sorted_used[:-1])
            ]
            target = model(bd_full, pm_batched, covariates=cov_t)
        else:
            target = model(bd_full, covariates=cov_t)
    target = target.detach()

    for epoch in range(epochs):
        masked = masked_features()
        bd_m = []
        for L in used_levels:
            x = masked[L]
            B_, N_, _ = x.shape
            bd = make_level_batch(
                x.detach().cpu().numpy(),
                levels[L].edge_index,
                levels[L].edge_weight,
                device,
            )
            bd["x"] = x.reshape(B_ * N_, -1)
            bd_m.append(bd)
        if is_unet:
            sorted_used = sorted(used_levels)
            pm_batched = [
                make_pool_map_batch(pm, levels[Lf].coords.shape[0], 1, device)
                for pm, Lf in zip(pool_maps_list, sorted_used[:-1])
            ]
            pred = model(bd_m, pm_batched, covariates=cov_t)
        else:
            pred = model(bd_m, covariates=cov_t)
        # Stay close to original prediction, with sparsity prior on the masks.
        loss_fit = ((pred - target) ** 2).mean()
        loss_sparse = 0.05 * (torch.sigmoid(node_logits).mean() + torch.sigmoid(feat_logits).mean())
        loss = loss_fit + loss_sparse
        optim.zero_grad(); loss.backward(); optim.step()

    return (
        torch.sigmoid(node_logits).detach().cpu().numpy(),
        torch.sigmoid(feat_logits).detach().cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Save importance back to VTK for ParaView
# ---------------------------------------------------------------------------


def write_importance_vtk(
    source_vtk: vtk.vtkPolyData,
    node_importance: np.ndarray,
    array_name: str,
    out_path,
) -> None:
    """Attach a per-node importance scalar to the source polydata and save."""
    arr = numpy_to_vtk(node_importance.astype(np.float32), deep=True)
    arr.SetName(array_name)
    source_vtk.GetPointData().AddArray(arr)
    source_vtk.GetPointData().SetActiveScalars(array_name)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(source_vtk)
    writer.SetFileTypeToBinary()
    writer.Write()
