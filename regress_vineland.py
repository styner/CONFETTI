#!/usr/bin/env python3
"""Multi-output Vineland regression: nested CV + interpretability.

Identical backbone and harness as classify_ds.py; the only differences are:
  - Targets: 5 V06 Vineland standard scores (composite + 4 subscales)
  - Loss: SmoothL1 with per-target NaN masking
  - Baselines: Ridge regression + per-target XGBoostRegressor
  - Metrics: Pearson r and MAE per target
  - Interpretability: per-target attribution via integrated_gradients(target_idx=k)
    plus per-tract permutation importance using MAE (lower-is-better -> we
    report importance as the rise in MAE rather than drop in AUC).

CLI quickstart:
  python regress_vineland.py
  python regress_vineland.py --model unet --imputation knn --quick
"""

from __future__ import annotations

# Avoid the macOS OpenMP runtime conflict between PyTorch, XGBoost, and
# scikit-learn (which each bundle their own libomp). Must be set BEFORE the
# offending libraries import, so we keep this at the top before any other
# imports. Single-threaded OMP also avoids occasional deadlocks under the
# same fork. See: OMP: Error #179 / pthread_mutex_init.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ds_pipeline.baselines import (
    RidgeRegressor,
    XGBoostRegressor,
    per_tract_mean_features,
)
from ds_pipeline.config import (
    DEFAULT_INNER_FOLDS,
    DEFAULT_LEVEL_BASE,
    DEFAULT_LEVELS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTER_REPEATS,
    DEFAULT_SEED,
    DEFAULT_SIREN_EPOCHS,
    DEFAULT_SIREN_OMEGA0,
    PROPERTIES,
    VINELAND_SHORT_NAMES,
)
from ds_pipeline.cv import (
    TrainConfig,
    nested_cv_regression,
)
from ds_pipeline.config import COVARIATE_NAMES as _COVARIATE_NAMES  # noqa: F401
from ds_pipeline.data import (
    build_covariate_matrix,
    load_levels,
    load_v06_subjects,
    pool_maps_from_levels,
    select_subjects,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--level-base", default=DEFAULT_LEVEL_BASE)
    p.add_argument("--labels-csv", type=Path, default=Path("Data_IBISEP_DS/EP_DS_DWI_data_V06.csv"))
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--model", choices=("gcn", "multi", "unet"), default="unet")
    p.add_argument("--imputation", choices=("knn", "siren"), default="knn")
    p.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    p.add_argument("--repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false")
    p.set_defaults(use_batchnorm=True)
    p.add_argument("--no-curves", dest="save_curves", action="store_false")
    p.set_defaults(save_curves=True)
    p.add_argument("--curves-dir", type=Path, default=None)
    p.add_argument("--no-covariates", dest="use_covariates", action="store_false")
    p.set_defaults(use_covariates=True)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--results-dir", type=Path, default=Path("vineland_results"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--siren-epochs", type=int, default=DEFAULT_SIREN_EPOCHS)
    p.add_argument("--siren-omega0", type=float, default=DEFAULT_SIREN_OMEGA0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.repeats = 1; args.outer_folds = 2; args.inner_folds = 2; args.epochs = 30

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading levels {args.levels}...")
    levels = load_levels(
        args.data_dir, tuple(args.levels), base=args.level_base, properties=PROPERTIES
    )
    pool_maps = pool_maps_from_levels(levels)

    print(f"Loading labels from {args.labels_csv}")
    df = load_v06_subjects(args.labels_csv)
    subject_ids = select_subjects(levels, df, require_label=False, require_any_vineland=True)
    if not subject_ids:
        raise SystemExit("No subjects with both VTK data and any Vineland score.")
    target_cols = [f"vineland_{n}" for n in VINELAND_SHORT_NAMES]
    y = df.loc[subject_ids, target_cols].to_numpy(dtype=np.float32)
    covariates_all = build_covariate_matrix(df, subject_ids) if args.use_covariates else None
    if covariates_all is not None:
        print(f"  covariates: shape={covariates_all.shape}, NaN cells={int(np.isnan(covariates_all).sum())}")
    n_with_any = int((~np.isnan(y).all(axis=1)).sum())
    print(f"  regression subjects: {len(subject_ids)} ({n_with_any} have >=1 Vineland score)")
    for t, name in enumerate(VINELAND_SHORT_NAMES):
        n_t = int((~np.isnan(y[:, t])).sum())
        print(f"    target {name}: n={n_t}, mean={np.nanmean(y[:, t]):.1f}, std={np.nanstd(y[:, t]):.1f}")

    # --- Baseline plug-ins ---
    L0 = min(levels)
    ld0 = levels[L0]

    def baseline_features(imputed, ids):
        X, _ = per_tract_mean_features(ld0, ids, imputed, properties=PROPERTIES)
        return X

    baseline_fns = {
        "ridge": RidgeRegressor(alpha=1.0, seed=args.seed).fit_predict,
        "xgboost": XGBoostRegressor(seed=args.seed).fit_predict,
    }

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        use_batchnorm=args.use_batchnorm, device=args.device,
    )
    curves_dir = (args.curves_dir or (args.results_dir / "curves")) if args.save_curves else None
    if curves_dir is not None:
        curves_dir.mkdir(parents=True, exist_ok=True)
    results = nested_cv_regression(
        levels=levels,
        pool_maps_by_level=pool_maps,
        subject_ids=subject_ids,
        y=y,
        model_name=args.model,
        target_names=VINELAND_SHORT_NAMES,
        baseline_fns=baseline_fns,
        baseline_feature_extractor=baseline_features,
        imputation=args.imputation,
        siren_cfg={"epochs": args.siren_epochs, "omega_0": args.siren_omega0, "seed": args.seed},
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        repeats=args.repeats, seed=args.seed,
        train_cfg=train_cfg,
        properties=PROPERTIES,
        curves_dir=curves_dir,
        covariates=covariates_all,
    )
    results.to_csv(args.results_dir / f"vineland_cv_results_{args.model}_{args.imputation}.csv", index=False)
    print("\nSummary (mean +/- std) per (method, target):")
    summary = results.groupby(["method", "target"]).agg(
        r_mean=("pearson_r", "mean"), r_std=("pearson_r", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n=("pearson_r", "count"),
    ).sort_index()
    print(summary.to_string())
    summary.to_csv(args.results_dir / f"vineland_cv_summary_{args.model}_{args.imputation}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
