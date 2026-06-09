#!/usr/bin/env python3
"""Experiment 3: per-property OOF perm-imp restricted to AC_olfactory (tract 0).

For each architecture (gcn @ L3, multi @ L2+L3, unet @ L2-L4) under the same
nested 5x3 stratified CV protocol used in classify_ds.py:

  For each fold:
    1. Train the GNN on the training fold (same hyperparameters as final runs)
    2. For each property in (AD, FA, NDI, ODI, RD):
         shuffle ONLY that property's column across subjects at tract-0 nodes
         compute AUC drop -> store per-fold value
  Aggregate across folds -> mean +/- std per property.

This tells you which biological modality (AD/FA/NDI/ODI/RD) drives the GNN's
AC_olfactory signal. NDI/ODI dominance would align with the NODDI-based
"neurite-density / orientation-dispersion" story for DS; FA/RD dominance
would point at classical fiber-coherence / myelin signatures.

Uses --skip-interpret and --no-explain-oof equivalents internally; only
trains and runs the focused perm step. Wall time per model: roughly
the CV cost (~5-30 min) plus a few seconds of permutation per fold.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ds_pipeline.config import (
    COVARIATE_NAMES,
    DEFAULT_LEVEL_BASE,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTER_REPEATS,
    DEFAULT_SEED,
    PROPERTIES,
)
from ds_pipeline.cv import (
    GNNTrainer,
    TrainConfig,
    _apply_covariate_stats,
    _apply_property_stats,
    _build_inputs,
    _covariate_train_stats,
    _maybe_impute,
    _per_property_train_stats,
    _propagate_imputed_to_levels,
)
from ds_pipeline.data import (
    build_covariate_matrix,
    load_levels,
    load_v06_subjects,
    pool_maps_from_levels,
    select_subjects,
)
from ds_pipeline.imputation import KnnImputer
from ds_pipeline.interpret import _predict
from ds_pipeline.models import build_model


def per_property_perm_at_tract(
    model,
    feats_used,
    levels,
    used_levels,
    pool_maps_list,
    y_te,
    cov_te,
    is_unet,
    device,
    tract_id,
    n_repeats,
    rng,
):
    """For each property p, shuffle feats_used[primary_L][:, tract_nodes, p]
    across subjects, measure base_auc - permuted_auc, return dict[prop -> mean_drop]."""
    L0 = used_levels[0]
    fiber_label = levels[L0].fiber_label
    node_idx = np.where(fiber_label == tract_id)[0]
    base = _predict(model, feats_used, levels, used_levels, pool_maps_list,
                    is_unet, device, covariates=cov_te)
    base_auc = float(roc_auc_score(y_te, 1.0 / (1.0 + np.exp(-base.squeeze(-1)))))
    out = {}
    for p_idx, prop in enumerate(PROPERTIES):
        drops = []
        for _ in range(n_repeats):
            shuffled = {L: feats_used[L].copy() for L in used_levels}
            x = shuffled[L0]
            S = x.shape[0]
            perm = rng.permutation(S)
            x[:, node_idx, p_idx] = x[perm][:, node_idx, p_idx]
            shuffled[L0] = x
            logits = _predict(model, shuffled, levels, used_levels, pool_maps_list,
                              is_unet, device, covariates=cov_te)
            auc = float(roc_auc_score(y_te, 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))))
            drops.append(base_auc - auc)
        out[prop] = float(np.mean(drops))
    return out


def run_one(args, model_name: str, levels_int: tuple[int, ...], dropout: float):
    print(f"\n=== {model_name} (levels={levels_int} dropout={dropout}) ===")
    levels = load_levels(Path("OutputMerged"), levels_int,
                        base=DEFAULT_LEVEL_BASE, properties=PROPERTIES)
    pool_maps = pool_maps_from_levels(levels)
    label_df = load_v06_subjects(Path("Data_IBISEP_DS/EP_DS_DWI_data_V06.csv"))
    sids = select_subjects(levels, label_df, require_label=True)
    y = np.array([int(label_df.loc[s, "label"]) for s in sids])
    cov = build_covariate_matrix(label_df, sids)

    # kNN-impute once (no fold leakage).
    L0 = min(levels); ld0 = levels[L0]
    knn_cached = KnnImputer(k=5).impute(ld0.coords, ld0.property_values, PROPERTIES)

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=1e-3, weight_decay=1e-3,
        hidden_dim=64, dropout=dropout, use_batchnorm=True,
        device="auto",
    )
    device = train_cfg.select_device()
    rng = np.random.default_rng(args.seed)

    sorted_levels = sorted(levels)
    used_levels = ([sorted_levels[-1]] if model_name == "gcn" else sorted_levels)
    sorted_used = sorted(used_levels)
    pool_maps_list = (
        [pool_maps[(L, Lp1)] for L, Lp1 in zip(sorted_used[:-1], sorted_used[1:])]
        if len(used_levels) > 1 and model_name == "unet" else []
    )

    per_prop_drops: dict[str, list[float]] = defaultdict(list)

    fold_count = 0
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True,
                              random_state=args.seed + rep)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(sids, y)):
            train_ids = [sids[i] for i in tr_idx]
            test_ids = [sids[i] for i in te_idx]
            y_tr = y[tr_idx]; y_te = y[te_idx]
            if len(np.unique(y_te)) < 2:
                continue
            fold_count += 1

            # Per-fold imputed values: cached kNN (no leakage).
            imputed_L0 = _maybe_impute(levels, train_ids, "knn", knn_cached,
                                       {}, PROPERTIES)
            imputed_per_level = _propagate_imputed_to_levels(levels, imputed_L0, pool_maps)

            # Per-fold covariate stats.
            c_mu, c_sd = _covariate_train_stats(cov, tr_idx)
            cov_full = _apply_covariate_stats(cov, c_mu, c_sd)
            cov_tr_outer = cov_full[tr_idx]
            cov_te = cov_full[te_idx]

            # Inner split for early stopping.
            inner_kf = StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=args.seed + rep + 17)
            it = iter(inner_kf.split(train_ids, y_tr))
            tr_inner_idx, va_inner_idx = next(it)
            inner_train_ids = [train_ids[i] for i in tr_inner_idx]
            inner_val_ids = [train_ids[i] for i in va_inner_idx]
            cov_tr_inner = cov_tr_outer[tr_inner_idx]
            cov_va_inner = cov_tr_outer[va_inner_idx]
            y_inner_train = y_tr[tr_inner_idx]
            y_inner_val = y_tr[va_inner_idx]

            trainer = GNNTrainer(model_name, train_cfg, "classification")
            _, _, info = trainer.fit_predict(
                levels=levels,
                pool_maps_by_level=pool_maps,
                train_subject_ids=inner_train_ids,
                val_subject_ids=inner_val_ids,
                test_subject_ids=test_ids,
                imputed_per_level=imputed_per_level,
                y_train=y_inner_train,
                y_val=y_inner_val,
                properties=PROPERTIES,
                n_outputs=1,
                covariates_train=cov_tr_inner,
                covariates_val=cov_va_inner,
                covariates_test=cov_te,
            )

            # Rebuild model from info["model_state"] for the perm pass.
            in_dim = len(PROPERTIES) + 4
            n_lv = 1 if len(used_levels) == 1 else len(used_levels)
            model = build_model(
                model_name, in_dim=in_dim, n_levels=n_lv, out_dim=1,
                hidden_dim=64, dropout=dropout, use_batchnorm=True,
                n_covariates=cov.shape[1],
            ).to(device)
            model.load_state_dict(info["model_state"])
            model.eval()

            # Build test features with training-fold property stats.
            feats_test, _, _ = _build_inputs(levels, test_ids, imputed_per_level, PROPERTIES)
            _apply_property_stats(feats_test, info["prop_stats"], PROPERTIES)
            feats_used = {L: feats_test[L] for L in used_levels}

            is_unet = (model_name == "unet")
            drops = per_property_perm_at_tract(
                model, feats_used, levels, used_levels, pool_maps_list,
                y_te, cov_te, is_unet, device,
                tract_id=0, n_repeats=args.n_perm_repeats, rng=rng,
            )
            for p, v in drops.items():
                per_prop_drops[p].append(v)
            print(f"  fold {fold_count}: " +
                  "  ".join(f"{p}={v:+.4f}" for p, v in drops.items()))

    summary = {p: (float(np.mean(v)), float(np.std(v)), len(v))
               for p, v in per_prop_drops.items()}
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument("--repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-perm-repeats", type=int, default=5)
    args = p.parse_args()

    out_dir = Path("ds_results/validation_AC_olfactory")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [("gcn", (2, 3), 0.2),
               ("multi", (2, 3), 0.3),
               ("unet", (2, 3, 4), 0.2)]
    all_rows = []
    for name, lv, d in configs:
        s = run_one(args, name, lv, d)
        for prop, (mu, sd, n) in s.items():
            all_rows.append({"model": name, "property": prop,
                             "importance_mean": mu, "importance_std": sd, "n_folds": n})

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "per_property_perm_AC_olfactory.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 78)
    print("Per-property OOF perm-importance restricted to AC_olfactory (tract 0)")
    print("=" * 78)
    print(df.pivot(index="property", columns="model", values="importance_mean")
            .reindex(PROPERTIES).round(4).to_string())
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
