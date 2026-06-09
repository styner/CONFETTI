"""Cross-validation harness and GNN training loop.

Public API:
  nested_cv_classification()  -- runs nested 5x5 CV for the DS classification task
  nested_cv_regression()      -- runs nested 5x5 CV for multi-output Vineland
  TrainConfig                 -- training hyperparameters as a dataclass
  GNNTrainer                  -- one-fold training and prediction helper

Design notes:
  - One imputation per outer fold (kNN can be cached; SIREN needs per-fold).
  - Inner CV is for hyperparameter selection only; it reuses the outer-fold
    imputation. This is the standard "no test info leaks into training"
    convention while keeping the SIREN cost manageable.
  - Augmentation is implemented in `_augment_inputs`: Gaussian noise on the
    property channels + random per-tract dropout mask.
  - Metrics: ROC-AUC for classification; per-target Pearson r and MAE for
    multi-output regression.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import pearsonr

from .config import (
    DEFAULT_INNER_FOLDS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTER_REPEATS,
    DEFAULT_SEED,
    PROPERTIES,
    VINELAND_SHORT_NAMES,
)
from .data import (
    LevelData,
    build_subject_tensor_for_level,
    coordinate_bounds,
)
from .imputation import KnnImputer, SirenImputer
from .models import (
    HierarchicalGraphUNet,
    build_model,
    make_level_batch,
    make_pool_map_batch,
)


# ---------------------------------------------------------------------------
# Train config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for GNN training (per-fold)."""

    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.4
    use_batchnorm: bool = True         # BatchNorm1d after each GCNConv
    batch_size: int = 32               # not used if full-batch (default)
    full_batch: bool = True
    noise_std: float = 0.05            # Gaussian noise on property channels
    tract_dropout_p: float = 0.10      # per-epoch random tract mask
    early_stopping_patience: int = 150
    device: str = "auto"

    def select_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tensors per level: build once per fold
# ---------------------------------------------------------------------------


def _build_inputs(
    levels: dict[int, LevelData],
    subject_ids: list[str],
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
    properties: tuple[str, ...],
    center: dict[int, np.ndarray] | None = None,
    half: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build (level -> (S, N, F)) feature tensors, applying spatial coordinate
    normalization. If `center`/`half` are provided they are used (training-set
    stats); otherwise per-level stats are computed from this batch.

    `imputed_per_level[L]` must give property arrays sized to that level.
    """
    feats: dict[int, np.ndarray] = {}
    out_center: dict[int, np.ndarray] = {}
    out_half: dict[int, np.ndarray] = {}
    n_prop = len(properties)
    for L, ld in levels.items():
        x = build_subject_tensor_for_level(
            ld, subject_ids, imputed_per_level[L], properties
        )
        c, h = coordinate_bounds(ld) if center is None else (center[L], half[L])
        # Normalize the spatial block (last 4 features) per level.
        spatial = x[..., n_prop:]
        x[..., n_prop:] = (spatial - c) / h
        feats[L] = x
        out_center[L] = c
        out_half[L] = h
    return feats, out_center, out_half


def _per_property_train_stats(
    levels: dict[int, LevelData],
    train_subject_ids: list[str],
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
    properties: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    """Per-property mean/std over L0 imputed values of training subjects only.
    Used to z-score the property channels of every input tensor."""
    stats: dict[str, tuple[float, float]] = {}
    L0 = min(levels)
    imputed_L0 = imputed_per_level[L0]
    train_set = set(train_subject_ids)
    for prop in properties:
        chunks: list[np.ndarray] = []
        for sid, arr in imputed_L0[prop].items():
            if sid in train_set:
                chunks.append(arr)
        if not chunks:
            stats[prop] = (0.0, 1.0)
            continue
        cat = np.concatenate(chunks)
        mu = float(cat.mean())
        sd = float(cat.std()) or 1.0
        stats[prop] = (mu, sd)
    return stats


def _apply_property_stats(
    feats: dict[int, np.ndarray],
    prop_stats: dict[str, tuple[float, float]],
    properties: tuple[str, ...],
) -> None:
    """In-place z-score the property channels (the first len(properties) cols)."""
    n_prop = len(properties)
    for L, x in feats.items():
        for p_idx, prop in enumerate(properties):
            mu, sd = prop_stats[prop]
            x[..., p_idx] = (x[..., p_idx] - mu) / sd


def _apply_tract_ablation(
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
    levels: dict[int, LevelData],
    train_subject_ids: list[str],
    ablate_tracts: set[int],
    properties: tuple[str, ...],
) -> None:
    """Replace per-(property, subject) values at nodes belonging to the
    `ablate_tracts` with the training-fold mean (computed only over training
    subjects). The graph structure stays intact -- only the inputs are
    ablated -- so we can measure how much classification depends on those
    tracts' features. Modifies `imputed_per_level` in place.
    """
    if not ablate_tracts:
        return
    train_set = set(train_subject_ids)
    for L, ld in levels.items():
        node_mask = np.isin(ld.fiber_label, list(ablate_tracts))
        if not node_mask.any():
            continue
        for prop in properties:
            if prop not in imputed_per_level[L]:
                continue
            train_vals: list[np.ndarray] = []
            for sid, arr in imputed_per_level[L][prop].items():
                if sid in train_set:
                    train_vals.append(arr[node_mask])
            if not train_vals:
                continue
            train_mean = float(np.concatenate(train_vals).mean())
            for sid, arr in imputed_per_level[L][prop].items():
                new_arr = arr.copy()
                new_arr[node_mask] = train_mean
                imputed_per_level[L][prop][sid] = new_arr


def _covariate_train_stats(
    covariates: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std over training subjects ONLY, NaN-aware.

    Returns mean and std with std clamped to >=1 epsilon so divide-by-zero
    cannot happen on a constant column. Used to z-score and mean-impute
    covariates without leakage from val/test rows.
    """
    train_rows = covariates[train_idx]
    mu = np.nanmean(train_rows, axis=0)
    sd = np.nanstd(train_rows, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((np.isnan(sd)) | (sd < 1e-6), 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _apply_covariate_stats(
    covariates: np.ndarray, mu: np.ndarray, sd: np.ndarray
) -> np.ndarray:
    """Mean-impute missing values then z-score using (mu, sd). Returns a copy."""
    out = covariates.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        col_idx = np.where(nan_mask)[1]
        out[nan_mask] = mu[col_idx]
    return ((out - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# Training-curve plotting
# ---------------------------------------------------------------------------


def _save_training_curves(
    train_losses: list[float],
    val_metrics: list[float],
    out_path,
    title: str,
    val_label: str = "validation metric",
) -> None:
    """Save a 2-panel PNG: training loss (top) + validation metric (bottom).
    Marks the best-validation epoch with a vertical line. Silently skips if
    matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from pathlib import Path as _Path
    out_path = _Path(out_path)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(epochs, train_losses, color="steelblue", lw=1.2)
    axes[0].set_ylabel("training loss")
    axes[0].set_title(title, fontsize=10)
    axes[0].grid(alpha=0.3)

    val_arr = np.array(val_metrics, dtype=float)
    valid = ~np.isnan(val_arr)
    if valid.any():
        ep_v = epochs[valid]
        v = val_arr[valid]
        axes[1].plot(ep_v, v, color="darkorange", lw=1.2)
        best_local = int(np.argmax(v))
        axes[1].axvline(
            ep_v[best_local], color="seagreen", ls="--", lw=0.9,
            label=f"best={v[best_local]:.3f} @ ep{ep_v[best_local]}",
        )
        axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_ylabel(val_label)
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _save_summary_curves(
    all_train_losses: list[list[float]],
    all_val_metrics: list[list[float]],
    out_path,
    title: str,
    val_label: str = "validation metric",
) -> None:
    """Mean ± std training-loss and validation-metric curves across all folds.
    Aligned to the shortest fold's epoch count (handles early stopping)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from pathlib import Path as _Path
    out_path = _Path(out_path)

    n_folds = len(all_train_losses)
    if n_folds == 0:
        return
    min_epochs_t = min(len(t) for t in all_train_losses)
    min_epochs_v = min(len(v) for v in all_val_metrics)
    min_epochs = min(min_epochs_t, min_epochs_v)
    if min_epochs == 0:
        return
    tl = np.array([t[:min_epochs] for t in all_train_losses])
    vm = np.array([v[:min_epochs] for v in all_val_metrics])
    epochs = np.arange(1, min_epochs + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    tl_mu, tl_sd = np.nanmean(tl, axis=0), np.nanstd(tl, axis=0)
    axes[0].plot(epochs, tl_mu, color="steelblue", lw=1.5)
    axes[0].fill_between(epochs, tl_mu - tl_sd, tl_mu + tl_sd, color="steelblue", alpha=0.2)
    axes[0].set_ylabel("training loss (mean ± std)")
    axes[0].set_title(f"{title} ({n_folds} folds)", fontsize=10)
    axes[0].grid(alpha=0.3)

    vm_mu, vm_sd = np.nanmean(vm, axis=0), np.nanstd(vm, axis=0)
    axes[1].plot(epochs, vm_mu, color="darkorange", lw=1.5)
    axes[1].fill_between(epochs, vm_mu - vm_sd, vm_mu + vm_sd, color="darkorange", alpha=0.2)
    axes[1].set_ylabel(f"{val_label} (mean ± std)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GNN training and prediction
# ---------------------------------------------------------------------------


class GNNTrainer:
    """Train and predict for one fold using a chosen GNN architecture."""

    def __init__(self, model_name: str, train_cfg: TrainConfig, task: str):
        self.model_name = model_name
        self.cfg = train_cfg
        self.task = task  # "classification" or "regression"
        self.device = train_cfg.select_device()

    def _augment_inputs(
        self,
        feats: dict[int, torch.Tensor],
        n_prop: int,
        tract_label_per_level: dict[int, torch.Tensor],
        unique_tracts: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Apply Gaussian noise on property channels and random tract dropout
        (mask all nodes in a randomly-chosen tract per epoch). Tract dropout
        is applied identically across levels so the same tracts are masked."""
        if not self.training:
            return feats
        out: dict[int, torch.Tensor] = {}
        # Choose tracts to drop ONCE per call so all levels agree.
        if self.cfg.tract_dropout_p > 0 and unique_tracts.numel() > 0:
            n_drop = int(round(self.cfg.tract_dropout_p * unique_tracts.numel()))
            if n_drop > 0:
                perm = torch.randperm(unique_tracts.numel(), device=self.device)
                drop_set = set(unique_tracts[perm[:n_drop]].tolist())
            else:
                drop_set = set()
        else:
            drop_set = set()
        for L, x in feats.items():
            xb = x.clone()
            if self.cfg.noise_std > 0:
                noise = torch.randn_like(xb[..., :n_prop]) * self.cfg.noise_std
                xb[..., :n_prop] = xb[..., :n_prop] + noise
            if drop_set:
                tract_mask = torch.tensor(
                    [int(t) in drop_set for t in tract_label_per_level[L].tolist()],
                    dtype=torch.bool,
                    device=self.device,
                )
                xb[:, tract_mask, :n_prop] = 0.0
            out[L] = xb
        return out

    def fit_predict(
        self,
        levels: dict[int, LevelData],
        pool_maps_by_level: dict[tuple[int, int], np.ndarray],
        train_subject_ids: list[str],
        val_subject_ids: list[str],
        test_subject_ids: list[str],
        imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
        y_train: np.ndarray,
        y_val: np.ndarray | None,
        properties: tuple[str, ...] = PROPERTIES,
        n_outputs: int = 1,
        verbose: bool = False,
        covariates_train: np.ndarray | None = None,
        covariates_val: np.ndarray | None = None,
        covariates_test: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, dict]:
        """Train the model and return (test_predictions, best_val_metric, info)."""
        sorted_levels = sorted(levels.keys())
        # Property stats from training subjects only.
        prop_stats = _per_property_train_stats(
            levels, train_subject_ids, imputed_per_level, properties
        )

        # Build feature tensors per level.
        all_ids = train_subject_ids + val_subject_ids + test_subject_ids
        feats_np, center, half = _build_inputs(
            levels, all_ids, imputed_per_level, properties
        )
        _apply_property_stats(feats_np, prop_stats, properties)

        # Slice into train / val / test (preserve order in all_ids).
        n_tr = len(train_subject_ids)
        n_va = len(val_subject_ids)
        feats_tr: dict[int, np.ndarray] = {L: feats_np[L][:n_tr] for L in sorted_levels}
        feats_va: dict[int, np.ndarray] = (
            {L: feats_np[L][n_tr : n_tr + n_va] for L in sorted_levels} if n_va else None
        )
        feats_te: dict[int, np.ndarray] = {L: feats_np[L][n_tr + n_va :] for L in sorted_levels}

        device = self.device
        in_dim = feats_np[sorted_levels[0]].shape[-1]
        n_levels_for_model = (
            1 if self.model_name.lower() in ("gcn", "single", "single_level_gcn")
            else len(sorted_levels)
        )

        n_covariates = 0 if covariates_train is None else int(covariates_train.shape[1])

        model = build_model(
            self.model_name,
            in_dim=in_dim,
            n_levels=n_levels_for_model,
            out_dim=n_outputs,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            use_batchnorm=self.cfg.use_batchnorm,
            n_covariates=n_covariates,
        ).to(device)
        is_unet = isinstance(model, HierarchicalGraphUNet)
        is_single = n_levels_for_model == 1

        # Choose which levels go into model: SingleLevelGCN uses coarsest only.
        if is_single:
            used_levels = [sorted_levels[-1]]
        else:
            used_levels = sorted_levels

        # Static graph tensors (edge index + weights + tract labels per level).
        # These do NOT change per-subject, only per-level.
        tract_labels_t: dict[int, torch.Tensor] = {
            L: torch.as_tensor(levels[L].fiber_label, dtype=torch.long, device=device)
            for L in used_levels
        }
        unique_tracts = torch.unique(tract_labels_t[used_levels[0]])

        # Pool maps tensors (Graph U-Net only).
        if is_unet:
            sorted_used = sorted(used_levels)
            pm_t: list[torch.Tensor] = []
            B_dummy = 1  # to be re-built per batch with the actual B
            pool_pair_list = []
            for L, Lp1 in zip(sorted_used[:-1], sorted_used[1:]):
                pool_pair_list.append(pool_maps_by_level[(L, Lp1)])

        # Move per-subject features to device tensors.
        x_tr_t: dict[int, torch.Tensor] = {
            L: torch.from_numpy(feats_tr[L]).to(device) for L in used_levels
        }
        x_va_t = (
            {L: torch.from_numpy(feats_va[L]).to(device) for L in used_levels}
            if feats_va is not None
            else None
        )
        x_te_t: dict[int, torch.Tensor] = {
            L: torch.from_numpy(feats_te[L]).to(device) for L in used_levels
        }

        # Covariate device tensors (already z-scored by the caller).
        def _to_dev(arr: np.ndarray | None) -> torch.Tensor | None:
            if arr is None:
                return None
            return torch.as_tensor(arr, dtype=torch.float32, device=device)
        c_tr_t = _to_dev(covariates_train)
        c_va_t = _to_dev(covariates_val)
        c_te_t = _to_dev(covariates_test)

        # For regression, z-score targets using training-fold stats; undo
        # the normalization at prediction time. NaNs are preserved.
        if self.task == "regression":
            y_train_np = np.asarray(y_train, dtype=np.float32)
            y_mu = np.nanmean(y_train_np, axis=0)
            y_sd = np.nanstd(y_train_np, axis=0)
            y_sd = np.where(y_sd > 0, y_sd, 1.0)
            y_train_z = (y_train_np - y_mu) / y_sd
            y_tr_t = torch.as_tensor(y_train_z, dtype=torch.float32, device=device)
            if y_val is not None:
                y_val_z = (np.asarray(y_val, dtype=np.float32) - y_mu) / y_sd
                y_va_t = torch.as_tensor(y_val_z, dtype=torch.float32, device=device)
            else:
                y_va_t = None
        else:
            y_tr_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
            y_va_t = (
                torch.as_tensor(y_val, dtype=torch.float32, device=device)
                if y_val is not None
                else None
            )
            y_mu = y_sd = None

        if self.task == "classification":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.SmoothL1Loss(reduction="none")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        n_prop = len(properties)
        best_val = -math.inf
        best_state: dict | None = None
        patience = 0
        train_losses: list[float] = []
        val_metrics: list[float] = []

        for epoch in range(self.cfg.epochs):
            model.train()
            self.training = True

            # Build batched dict per level for training (with augmentation).
            x_aug = self._augment_inputs(x_tr_t, n_prop, tract_labels_t, unique_tracts)
            train_batches = []
            B = next(iter(x_aug.values())).shape[0]
            for L in used_levels:
                bd = make_level_batch(
                    x_aug[L].cpu().numpy(),
                    levels[L].edge_index,
                    levels[L].edge_weight,
                    device,
                )
                train_batches.append(bd)
            if is_unet:
                pm_batched = [
                    make_pool_map_batch(pm, levels[Lf].coords.shape[0], B, device)
                    for pm, Lf in zip(pool_pair_list, sorted_used[:-1])
                ]
                logits = model(train_batches, pm_batched, covariates=c_tr_t)
            else:
                logits = model(train_batches, covariates=c_tr_t)

            if self.task == "classification":
                logits = logits.squeeze(-1)
                loss = criterion(logits, y_tr_t)
            else:
                # Regression: mask NaN targets per-sample, per-output.
                pred = logits
                mask = ~torch.isnan(y_tr_t)
                err = criterion(pred, torch.where(mask, y_tr_t, torch.zeros_like(y_tr_t)))
                err = err * mask
                loss = err.sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

            # Validation
            if x_va_t is not None:
                model.eval()
                self.training = False
                with torch.no_grad():
                    val_batches = []
                    Bv = next(iter(x_va_t.values())).shape[0]
                    for L in used_levels:
                        bd = make_level_batch(
                            x_va_t[L].cpu().numpy(),
                            levels[L].edge_index,
                            levels[L].edge_weight,
                            device,
                        )
                        val_batches.append(bd)
                    if is_unet:
                        pm_val = [
                            make_pool_map_batch(pm, levels[Lf].coords.shape[0], Bv, device)
                            for pm, Lf in zip(pool_pair_list, sorted_used[:-1])
                        ]
                        val_logits = model(val_batches, pm_val, covariates=c_va_t)
                    else:
                        val_logits = model(val_batches, covariates=c_va_t)
                    val_metric = self._val_metric(val_logits, y_va_t)
                val_metrics.append(float(val_metric))
                if val_metric > best_val:
                    best_val = val_metric
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break
            else:
                val_metrics.append(float("nan"))

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.training = False
        with torch.no_grad():
            test_batches = []
            Bt = next(iter(x_te_t.values())).shape[0]
            for L in used_levels:
                bd = make_level_batch(
                    x_te_t[L].cpu().numpy(),
                    levels[L].edge_index,
                    levels[L].edge_weight,
                    device,
                )
                test_batches.append(bd)
            if is_unet:
                pm_test = [
                    make_pool_map_batch(pm, levels[Lf].coords.shape[0], Bt, device)
                    for pm, Lf in zip(pool_pair_list, sorted_used[:-1])
                ]
                test_logits = model(test_batches, pm_test, covariates=c_te_t)
            else:
                test_logits = model(test_batches, covariates=c_te_t)

        test_out = test_logits.cpu().numpy()
        if self.task == "regression" and y_mu is not None:
            test_out = test_out * y_sd + y_mu

        info = {
            "best_val_metric": float(best_val) if best_val > -math.inf else float("nan"),
            "prop_stats": prop_stats,
            "model_state": best_state if best_state is not None else model.state_dict(),
            "y_mu": y_mu, "y_sd": y_sd,
            "train_losses": train_losses,
            "val_metrics": val_metrics,
        }
        return test_out, float(best_val), info

    def _val_metric(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Returns a value where HIGHER is better."""
        if self.task == "classification":
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            y_np = y.cpu().numpy()
            if len(np.unique(y_np)) < 2:
                return float("nan")
            return float(roc_auc_score(y_np, probs))
        else:
            pred = logits.cpu().numpy()
            y_np = y.cpu().numpy()
            mask = ~np.isnan(y_np)
            if not mask.any():
                return float("nan")
            # Negative MAE as a higher-is-better score.
            return float(-np.mean(np.abs(pred[mask] - y_np[mask])))


# ---------------------------------------------------------------------------
# Outer nested CV
# ---------------------------------------------------------------------------


def _maybe_impute(
    levels: dict[int, LevelData],
    train_subject_ids: list[str],
    method: str,
    knn_cached: dict[str, dict[str, np.ndarray]] | None,
    siren_cfg: dict,
    properties: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Return imputed property arrays at L0 for the outer fold."""
    L0 = min(levels)
    ld = levels[L0]
    if method == "knn":
        if knn_cached is not None:
            return knn_cached
        imputer = KnnImputer(k=5)
        return imputer.impute(ld.coords, ld.property_values, properties)
    elif method == "siren":
        imputer = SirenImputer(**siren_cfg)
        return imputer.impute(
            ld.coords, ld.property_values, properties, train_subjects=train_subject_ids
        )
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def _propagate_imputed_to_levels(
    levels: dict[int, LevelData],
    imputed_finest: dict[str, dict[str, np.ndarray]],
    pool_maps_by_level: dict[tuple[int, int], np.ndarray],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """Project the finest-level imputed property arrays down through every
    coarser level using the precomputed pool maps.

    `imputed_finest` is sized to the *smallest loaded level*. For each
    consecutive (L, L+1) pair we project using pool_maps_by_level[(L, L+1)],
    which gives positions in the L-frame that correspond to L+1 nodes.
    """
    sorted_levels = sorted(levels.keys())
    L_finest = sorted_levels[0]
    out: dict[int, dict[str, dict[str, np.ndarray]]] = {L_finest: imputed_finest}
    for prev_L, this_L in zip(sorted_levels[:-1], sorted_levels[1:]):
        prev_arrays = out[prev_L]
        pm = pool_maps_by_level[(prev_L, this_L)]
        prop_map: dict[str, dict[str, np.ndarray]] = {}
        for prop, by_sub in prev_arrays.items():
            prop_map[prop] = {sid: arr[pm] for sid, arr in by_sub.items()}
        out[this_L] = prop_map
    return out


def _build_imputed_dict_for_all_levels(
    levels: dict[int, LevelData],
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    return imputed_per_level


def _perm_imp_per_fold(
    *,
    levels: dict[int, LevelData],
    pool_maps_by_level: dict[tuple[int, int], np.ndarray],
    test_ids: list[str],
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
    info: dict,
    train_cfg: TrainConfig,
    model_name: str,
    properties: tuple[str, ...],
    n_outputs: int,
    y_te: np.ndarray,
    cov_te: np.ndarray | None,
    perm_results: dict[str, dict[object, list[float]]],
    n_tract_repeats: int,
    n_node_repeats: int,
    n_cov_repeats: int,
    skip_node_perm: bool,
    seed: int,
    covariate_names: tuple[str, ...] | None,
) -> None:
    """Compute permutation importance on THIS fold's held-out test set, using
    the just-trained model. Appends per-(granularity, gid) per-fold values to
    `perm_results` for later aggregation. Distinct from the post-CV refit-all
    permutation importance which suffers from ceiling-effect overfit.
    """
    from .interpret import permutation_importance
    from .models import build_model

    if len(y_te) == 0 or len(np.unique(y_te)) < 2:
        return  # need 2 classes to compute a meaningful AUC drop

    device = train_cfg.select_device()
    sorted_levels = sorted(levels.keys())
    used_levels = (
        [sorted_levels[-1]]
        if model_name.lower() in ("gcn", "single", "single_level_gcn")
        else sorted_levels
    )
    in_dim = len(properties) + 4
    n_lv = 1 if len(used_levels) == 1 else len(used_levels)
    n_covariates = 0 if cov_te is None else int(cov_te.shape[1])
    model = build_model(
        model_name,
        in_dim=in_dim,
        n_levels=n_lv,
        out_dim=n_outputs,
        hidden_dim=train_cfg.hidden_dim,
        dropout=train_cfg.dropout,
        use_batchnorm=train_cfg.use_batchnorm,
        n_covariates=n_covariates,
    ).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()

    # Build per-level test feature tensors using training-fold property stats.
    feats_test, _, _ = _build_inputs(
        levels, test_ids, imputed_per_level, properties
    )
    _apply_property_stats(feats_test, info["prop_stats"], properties)
    feats_used = {L: feats_test[L] for L in used_levels}
    sorted_used = sorted(used_levels)
    pool_maps_list = (
        [pool_maps_by_level[(L, Lp1)] for L, Lp1 in zip(sorted_used[:-1], sorted_used[1:])]
        if len(used_levels) > 1 and model_name.lower() in ("unet", "graph_unet", "hierarchical")
        else []
    )

    # Tract-level
    tract_imp = permutation_importance(
        model, feats_used, y_te, levels, used_levels, pool_maps_list,
        granularity="tract", n_repeats=n_tract_repeats, seed=seed,
        device=device, covariates=cov_te,
    )
    for k, v in tract_imp.items():
        perm_results["tract"].setdefault(int(k), []).append(float(v))

    # Covariate-level (cheap)
    if cov_te is not None and covariate_names is not None:
        cov_imp = permutation_importance(
            model, feats_used, y_te, levels, used_levels, pool_maps_list,
            granularity="covariate", n_repeats=n_cov_repeats, seed=seed,
            device=device, covariates=cov_te, covariate_names=covariate_names,
        )
        for k, v in cov_imp.items():
            perm_results["covariate"].setdefault(str(k), []).append(float(v))

    # Node-level (most expensive; optional)
    if not skip_node_perm:
        node_imp = permutation_importance(
            model, feats_used, y_te, levels, used_levels, pool_maps_list,
            granularity="node", n_repeats=n_node_repeats, seed=seed,
            device=device, covariates=cov_te,
        )
        for k, v in node_imp.items():
            perm_results["node"].setdefault(int(k), []).append(float(v))


def _explain_ds_in_fold(
    *,
    ds_oof: list[tuple[int, str]],
    test_ids: list[str],
    probs: np.ndarray,
    levels: dict[int, LevelData],
    pool_maps_by_level: dict[tuple[int, int], np.ndarray],
    imputed_per_level: dict[int, dict[str, dict[str, np.ndarray]]],
    info: dict,
    train_cfg: TrainConfig,
    model_name: str,
    properties: tuple[str, ...],
    n_outputs: int,
    repeat: int,
    fold_idx: int,
    explanations: dict[str, list[dict]],
    epochs: int,
    covariates_test: np.ndarray | None = None,
) -> None:
    """Run GNNExplainer on each held-out DS subject in a fold and append the
    soft node + feature masks to `explanations[subject_id]`.

    The model is rebuilt from `info["model_state"]` (i.e. the best-validation
    state from the inner split) so the explanation reflects the deployed model
    and the test subject was never in training.
    """
    from .interpret import gnn_explainer_sparse
    from .models import build_model

    device = train_cfg.select_device()
    sorted_levels = sorted(levels.keys())
    used_levels = (
        [sorted_levels[-1]]
        if model_name.lower() in ("gcn", "single", "single_level_gcn")
        else sorted_levels
    )
    in_dim = len(properties) + 4
    n_lv = 1 if len(used_levels) == 1 else len(used_levels)
    n_covariates = 0 if covariates_test is None else int(covariates_test.shape[1])
    model = build_model(
        model_name,
        in_dim=in_dim,
        n_levels=n_lv,
        out_dim=n_outputs,
        hidden_dim=train_cfg.hidden_dim,
        dropout=train_cfg.dropout,
        use_batchnorm=train_cfg.use_batchnorm,
        n_covariates=n_covariates,
    ).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()

    # Build per-level test feature tensors with training-fold property stats.
    feats_test, _, _ = _build_inputs(
        levels, test_ids, imputed_per_level, properties
    )
    _apply_property_stats(feats_test, info["prop_stats"], properties)
    feats_used = {L: feats_test[L] for L in used_levels}
    sorted_used = sorted(used_levels)
    pool_maps_list = (
        [pool_maps_by_level[(L, Lp1)] for L, Lp1 in zip(sorted_used[:-1], sorted_used[1:])]
        if len(used_levels) > 1 and model_name.lower() in ("unet", "graph_unet", "hierarchical")
        else []
    )

    L_primary = used_levels[0]
    for local_idx, sid in ds_oof:
        cov_one = (
            covariates_test[local_idx : local_idx + 1]
            if covariates_test is not None
            else None
        )
        node_mask, feat_mask = gnn_explainer_sparse(
            model,
            feats_used,
            local_idx,
            levels,
            used_levels,
            pool_maps_list,
            epochs=epochs,
            device=device,
            covariates=cov_one,
        )
        explanations.setdefault(sid, []).append({
            "fold": fold_idx,
            "repeat": repeat,
            "score": float(probs[local_idx]),
            "level": L_primary,
            "node_mask": node_mask,
            "feature_mask": feat_mask,
        })


def nested_cv_classification(
    levels: dict[int, LevelData],
    pool_maps_by_level: dict[tuple[int, int], np.ndarray],
    subject_ids: list[str],
    y: np.ndarray,
    model_name: str,
    baseline_fns: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] | None = None,
    baseline_feature_extractor: Callable[[dict[str, dict[str, np.ndarray]], list[str]], np.ndarray] | None = None,
    imputation: str = "knn",
    siren_cfg: dict | None = None,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    repeats: int = DEFAULT_OUTER_REPEATS,
    seed: int = DEFAULT_SEED,
    train_cfg: TrainConfig | None = None,
    properties: tuple[str, ...] = PROPERTIES,
    n_outputs: int = 1,
    explain_ds_oof: bool = False,
    explain_epochs: int = 200,
    curves_dir: "Path | None" = None,
    covariates: np.ndarray | None = None,
    covariate_names: tuple[str, ...] | None = None,
    perm_imp_oof: bool = False,
    perm_imp_skip_node: bool = False,
    perm_imp_tract_repeats: int = 3,
    perm_imp_node_repeats: int = 2,
    perm_imp_cov_repeats: int = 5,
    ablate_tracts: set[int] | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run nested CV for binary classification.

    Returns:
      (metrics_df, explanations) where
        metrics_df  -- tidy DataFrame with one row per (repeat, fold, method);
                       methods include each baseline name + 'gnn:<model>'.
        explanations -- {subject_id: [ {fold, repeat, score, node_mask,
                       feature_mask, level} ... ]}. Empty if `explain_ds_oof`
                       is False.

    When `explain_ds_oof` is True, GNNExplainer is run for each DS-positive
    subject when it appears in the held-out test fold (so the explanation is
    produced by a model that NEVER saw that subject in training). Outputs are
    per-node and per-feature soft masks.
    """
    siren_cfg = siren_cfg or {}
    train_cfg = train_cfg or TrainConfig()
    rows = []
    explanations: dict[str, list[dict]] = {}
    train_loss_history: list[list[float]] = []
    val_metric_history: list[list[float]] = []
    perm_results_per_fold: dict[str, dict[object, list[float]]] = {
        "tract": {}, "node": {}, "covariate": {},
    }
    knn_cached: dict[str, dict[str, np.ndarray]] | None = None
    if imputation == "knn":
        # Compute once (per-subject -> no fold leakage).
        L0 = min(levels)
        ld = levels[L0]
        knn_cached = KnnImputer(k=5).impute(ld.coords, ld.property_values, properties)

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed + rep)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(subject_ids, y)):
            train_ids = [subject_ids[i] for i in tr_idx]
            test_ids = [subject_ids[i] for i in te_idx]
            y_tr = y[tr_idx]; y_te = y[te_idx]

            t0 = time.time()
            imputed_L0 = _maybe_impute(
                levels, train_ids, imputation, knn_cached, siren_cfg, properties
            )
            imputed_per_level = _propagate_imputed_to_levels(levels, imputed_L0, pool_maps_by_level)
            if ablate_tracts:
                _apply_tract_ablation(
                    imputed_per_level, levels, train_ids, ablate_tracts, properties
                )
            imp_secs = time.time() - t0

            # Training-fold covariate stats (mean for imputation + sd for z-score).
            cov_tr = cov_te = None
            if covariates is not None:
                c_mu, c_sd = _covariate_train_stats(covariates, tr_idx)
                cov_full = _apply_covariate_stats(covariates, c_mu, c_sd)
                cov_tr = cov_full[tr_idx]
                cov_te = cov_full[te_idx]

            # Baselines (per-tract means computed at L0, optionally with covariates).
            if baseline_fns and baseline_feature_extractor is not None:
                X_full = baseline_feature_extractor(imputed_L0, train_ids + test_ids)
                if cov_tr is not None:
                    cov_concat = np.concatenate([cov_tr, cov_te], axis=0)
                    X_full = np.concatenate([X_full, cov_concat], axis=1)
                X_tr = X_full[: len(train_ids)]
                X_te = X_full[len(train_ids):]
                for name, fn in baseline_fns.items():
                    try:
                        proba = fn(X_tr, y_tr, X_te)
                        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) == 2 else float("nan")
                        f1 = f1_score(y_te, (proba >= 0.5).astype(int))
                    except Exception as exc:
                        auc, f1 = float("nan"), float("nan")
                        proba = None
                        print(f"  baseline {name} failed: {exc}")
                    rows.append({
                        "repeat": rep, "fold": fold_idx, "method": name,
                        "auc": auc, "f1": f1,
                    })

            # GNN inner split for early stopping.
            inner_kf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + rep + 17)
            it = iter(inner_kf.split(train_ids, y_tr))
            tr_inner_idx, va_inner_idx = next(it)
            inner_train_ids = [train_ids[i] for i in tr_inner_idx]
            inner_val_ids = [train_ids[i] for i in va_inner_idx]
            y_inner_train = y_tr[tr_inner_idx]
            y_inner_val = y_tr[va_inner_idx]
            cov_tr_inner = cov_tr[tr_inner_idx] if cov_tr is not None else None
            cov_va_inner = cov_tr[va_inner_idx] if cov_tr is not None else None

            trainer = GNNTrainer(model_name, train_cfg, "classification")
            test_logits, val_metric, info = trainer.fit_predict(
                levels=levels,
                pool_maps_by_level=pool_maps_by_level,
                train_subject_ids=inner_train_ids,
                val_subject_ids=inner_val_ids,
                test_subject_ids=test_ids,
                imputed_per_level=imputed_per_level,
                y_train=y_inner_train,
                y_val=y_inner_val,
                properties=properties,
                n_outputs=n_outputs,
                covariates_train=cov_tr_inner,
                covariates_val=cov_va_inner,
                covariates_test=cov_te,
            )
            probs = 1.0 / (1.0 + np.exp(-test_logits.squeeze(-1)))
            auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) == 2 else float("nan")
            f1 = f1_score(y_te, (probs >= 0.5).astype(int))
            rows.append({
                "repeat": rep, "fold": fold_idx, "method": f"gnn:{model_name}",
                "auc": auc, "f1": f1,
                "val_auc": val_metric,
                "impute_secs": imp_secs,
            })
            print(
                f"[rep={rep} fold={fold_idx}] {model_name}: "
                f"auc={auc:.3f}, f1={f1:.3f}, val_auc={val_metric:.3f}, impute={imp_secs:.0f}s"
            )

            train_loss_history.append(info.get("train_losses", []))
            val_metric_history.append(info.get("val_metrics", []))
            if curves_dir is not None:
                _save_training_curves(
                    info.get("train_losses", []),
                    info.get("val_metrics", []),
                    curves_dir / f"{model_name}_rep{rep}_fold{fold_idx}.png",
                    title=(
                        f"{model_name}  rep={rep} fold={fold_idx}  "
                        f"(dropout={train_cfg.dropout}, epochs={train_cfg.epochs})"
                    ),
                    val_label="validation AUC",
                )

            if perm_imp_oof:
                _perm_imp_per_fold(
                    levels=levels,
                    pool_maps_by_level=pool_maps_by_level,
                    test_ids=test_ids,
                    imputed_per_level=imputed_per_level,
                    info=info,
                    train_cfg=train_cfg,
                    model_name=model_name,
                    properties=properties,
                    n_outputs=n_outputs,
                    y_te=y_te,
                    cov_te=cov_te,
                    perm_results=perm_results_per_fold,
                    n_tract_repeats=perm_imp_tract_repeats,
                    n_node_repeats=perm_imp_node_repeats,
                    n_cov_repeats=perm_imp_cov_repeats,
                    skip_node_perm=perm_imp_skip_node,
                    seed=seed + rep * 1000 + fold_idx,
                    covariate_names=covariate_names,
                )

            if explain_ds_oof:
                ds_oof = [
                    (local_idx, sid)
                    for local_idx, sid in enumerate(test_ids)
                    if int(y_te[local_idx]) == 1
                ]
                if ds_oof:
                    _explain_ds_in_fold(
                        ds_oof=ds_oof,
                        test_ids=test_ids,
                        probs=probs,
                        levels=levels,
                        pool_maps_by_level=pool_maps_by_level,
                        imputed_per_level=imputed_per_level,
                        info=info,
                        train_cfg=train_cfg,
                        model_name=model_name,
                        properties=properties,
                        n_outputs=n_outputs,
                        repeat=rep,
                        fold_idx=fold_idx,
                        explanations=explanations,
                        epochs=explain_epochs,
                        covariates_test=cov_te,
                    )

    if curves_dir is not None and train_loss_history:
        _save_summary_curves(
            train_loss_history,
            val_metric_history,
            curves_dir / f"{model_name}_summary.png",
            title=(
                f"{model_name}  all folds  "
                f"(dropout={train_cfg.dropout}, epochs={train_cfg.epochs})"
            ),
            val_label="validation AUC",
        )

    # Aggregate per-fold OOF permutation importance (mean / std across folds).
    perm_imp_oof_summary: dict[str, dict] = {}
    for granularity, gid_to_values in perm_results_per_fold.items():
        if not gid_to_values:
            continue
        perm_imp_oof_summary[granularity] = {
            gid: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_folds": len(vals),
                "per_fold": list(vals),
            }
            for gid, vals in gid_to_values.items()
        }

    return pd.DataFrame(rows), explanations, perm_imp_oof_summary


def nested_cv_regression(
    levels: dict[int, LevelData],
    pool_maps_by_level: dict[tuple[int, int], np.ndarray],
    subject_ids: list[str],
    y: np.ndarray,
    model_name: str,
    target_names: tuple[str, ...] = VINELAND_SHORT_NAMES,
    baseline_fns: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] | None = None,
    baseline_feature_extractor: Callable[[dict[str, dict[str, np.ndarray]], list[str]], np.ndarray] | None = None,
    imputation: str = "knn",
    siren_cfg: dict | None = None,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    repeats: int = DEFAULT_OUTER_REPEATS,
    seed: int = DEFAULT_SEED,
    train_cfg: TrainConfig | None = None,
    properties: tuple[str, ...] = PROPERTIES,
    curves_dir: "Path | None" = None,
    covariates: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run nested CV for multi-output regression (Vineland).

    y has shape (N, K) with NaN for missing entries. Metrics per (repeat, fold,
    method, target): Pearson r, Spearman r, MAE.
    """
    siren_cfg = siren_cfg or {}
    train_cfg = train_cfg or TrainConfig()
    rows = []
    train_loss_history: list[list[float]] = []
    val_metric_history: list[list[float]] = []
    knn_cached = None
    if imputation == "knn":
        L0 = min(levels)
        ld = levels[L0]
        knn_cached = KnnImputer(k=5).impute(ld.coords, ld.property_values, properties)

    n_targets = y.shape[1]

    for rep in range(repeats):
        # Use KFold (not StratifiedKFold) for regression; stratify by class if
        # provided externally. Subjects with all-NaN y get dropped.
        valid = ~np.isnan(y).all(axis=1)
        ids_valid = [sid for sid, ok in zip(subject_ids, valid) if ok]
        y_valid = y[valid]
        cov_valid = covariates[valid] if covariates is not None else None
        kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed + rep)
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(ids_valid)):
            train_ids = [ids_valid[i] for i in tr_idx]
            test_ids = [ids_valid[i] for i in te_idx]
            y_tr = y_valid[tr_idx]; y_te = y_valid[te_idx]

            t0 = time.time()
            imputed_L0 = _maybe_impute(
                levels, train_ids, imputation, knn_cached, siren_cfg, properties
            )
            imputed_per_level = _propagate_imputed_to_levels(levels, imputed_L0, pool_maps_by_level)
            imp_secs = time.time() - t0

            cov_tr = cov_te = None
            if cov_valid is not None:
                c_mu, c_sd = _covariate_train_stats(cov_valid, tr_idx)
                cov_full = _apply_covariate_stats(cov_valid, c_mu, c_sd)
                cov_tr = cov_full[tr_idx]
                cov_te = cov_full[te_idx]

            # Baselines (per-tract means computed at L0).
            if baseline_fns and baseline_feature_extractor is not None:
                X_full = baseline_feature_extractor(imputed_L0, train_ids + test_ids)
                if cov_tr is not None:
                    cov_concat = np.concatenate([cov_tr, cov_te], axis=0)
                    X_full = np.concatenate([X_full, cov_concat], axis=1)
                X_tr = X_full[: len(train_ids)]
                X_te = X_full[len(train_ids):]
                for name, fn in baseline_fns.items():
                    try:
                        pred = fn(X_tr, y_tr, X_te)
                    except Exception as exc:
                        print(f"  baseline {name} failed: {exc}")
                        continue
                    for t, tname in enumerate(target_names):
                        mask = ~np.isnan(y_te[:, t])
                        if mask.sum() < 3:
                            continue
                        r = float(pearsonr(pred[mask, t], y_te[mask, t])[0])
                        mae = float(mean_absolute_error(y_te[mask, t], pred[mask, t]))
                        rows.append({
                            "repeat": rep, "fold": fold_idx, "method": name,
                            "target": tname, "pearson_r": r, "mae": mae,
                        })

            # GNN inner split for early stopping.
            inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + rep + 17)
            it = iter(inner_kf.split(train_ids))
            tr_inner_idx, va_inner_idx = next(it)
            inner_train_ids = [train_ids[i] for i in tr_inner_idx]
            inner_val_ids = [train_ids[i] for i in va_inner_idx]
            cov_tr_inner = cov_tr[tr_inner_idx] if cov_tr is not None else None
            cov_va_inner = cov_tr[va_inner_idx] if cov_tr is not None else None

            trainer = GNNTrainer(model_name, train_cfg, "regression")
            test_pred, val_metric, info = trainer.fit_predict(
                levels=levels,
                pool_maps_by_level=pool_maps_by_level,
                train_subject_ids=inner_train_ids,
                val_subject_ids=inner_val_ids,
                test_subject_ids=test_ids,
                imputed_per_level=imputed_per_level,
                y_train=y_tr[tr_inner_idx],
                y_val=y_tr[va_inner_idx],
                properties=properties,
                n_outputs=n_targets,
                covariates_train=cov_tr_inner,
                covariates_val=cov_va_inner,
                covariates_test=cov_te,
            )
            for t, tname in enumerate(target_names):
                mask = ~np.isnan(y_te[:, t])
                if mask.sum() < 3:
                    continue
                r = float(pearsonr(test_pred[mask, t], y_te[mask, t])[0])
                mae = float(mean_absolute_error(y_te[mask, t], test_pred[mask, t]))
                rows.append({
                    "repeat": rep, "fold": fold_idx, "method": f"gnn:{model_name}",
                    "target": tname, "pearson_r": r, "mae": mae,
                    "val_metric": val_metric,
                    "impute_secs": imp_secs,
                })
            print(
                f"[rep={rep} fold={fold_idx}] {model_name} regression: "
                f"val_metric={val_metric:.3f}, impute={imp_secs:.0f}s"
            )

            train_loss_history.append(info.get("train_losses", []))
            val_metric_history.append(info.get("val_metrics", []))
            if curves_dir is not None:
                _save_training_curves(
                    info.get("train_losses", []),
                    info.get("val_metrics", []),
                    curves_dir / f"{model_name}_rep{rep}_fold{fold_idx}.png",
                    title=(
                        f"{model_name}  rep={rep} fold={fold_idx}  "
                        f"(dropout={train_cfg.dropout}, epochs={train_cfg.epochs})"
                    ),
                    val_label="validation -MAE (higher better)",
                )

    if curves_dir is not None and train_loss_history:
        _save_summary_curves(
            train_loss_history,
            val_metric_history,
            curves_dir / f"{model_name}_summary.png",
            title=(
                f"{model_name}  all folds  "
                f"(dropout={train_cfg.dropout}, epochs={train_cfg.epochs})"
            ),
            val_label="validation -MAE (higher better)",
        )

    return pd.DataFrame(rows)
