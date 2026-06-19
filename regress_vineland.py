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
    TARGET_FAMILIES,
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
    p.add_argument("--no-perm-imp", dest="perm_imp_oof", action="store_false",
                   help="Skip OOF per-target permutation importance (tract + covariate + node).")
    p.set_defaults(perm_imp_oof=True)
    p.add_argument("--skip-node-perm", action="store_true",
                   help="Skip the slow per-node OOF perm-imp pass (keep tract + covariate).")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--results-dir", type=Path, default=Path("vineland_results"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--siren-epochs", type=int, default=DEFAULT_SIREN_EPOCHS)
    p.add_argument("--siren-omega0", type=float, default=DEFAULT_SIREN_OMEGA0)
    p.add_argument(
        "--target-family",
        choices=tuple(TARGET_FAMILIES.keys()),
        default="v24_all",
        help=(
            "Which regression target family to predict. "
            "'v06_vineland' = 5 V06 Vineland subscales (concurrent, original task). "
            "'v24_vineland' = 5 V24 Vineland subscales (prospective). "
            "'v24_bayley' = 3 V24 Bayley-4 scores (cog + EC + RC). "
            "'v24_all' = 8 combined V24 outcomes (default). "
            "All families use the same V06 imaging input."
        ),
    )
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

    # Pick the target family for this run.
    target_columns, target_short_names = TARGET_FAMILIES[args.target_family]
    print(f"Target family: {args.target_family}  ({len(target_short_names)} targets)")

    print(f"Loading labels from {args.labels_csv}")
    # The V06 (concurrent) family is handled by the load function's defaults;
    # any V24 family is wired through `extra_target_columns`.
    if args.target_family == "v06_vineland":
        df = load_v06_subjects(args.labels_csv)
        col_prefix = "vineland"
    else:
        df = load_v06_subjects(
            args.labels_csv,
            extra_target_columns=target_columns,
            extra_target_short_names=target_short_names,
        )
        col_prefix = "target"

    # Build the (S, K) target matrix; drop subjects with NO outcomes at all.
    target_cols = [f"{col_prefix}_{n}" for n in target_short_names]
    all_subject_ids = select_subjects(
        levels, df, require_label=False, require_any_vineland=False
    )
    y_full = df.loc[all_subject_ids, target_cols].to_numpy(dtype=np.float32)
    any_outcome = ~np.isnan(y_full).all(axis=1)
    subject_ids = [sid for sid, ok in zip(all_subject_ids, any_outcome) if ok]
    y = y_full[any_outcome]
    if not subject_ids:
        raise SystemExit(
            f"No subjects with VTK data AND any target in family '{args.target_family}'."
        )

    covariates_all = build_covariate_matrix(df, subject_ids) if args.use_covariates else None
    if covariates_all is not None:
        print(f"  covariates: shape={covariates_all.shape}, NaN cells={int(np.isnan(covariates_all).sum())}")
    print(f"  regression subjects: {len(subject_ids)} (have >=1 outcome in the family)")
    for t, name in enumerate(target_short_names):
        n_t = int((~np.isnan(y[:, t])).sum())
        print(
            f"    target {name}: n={n_t}, mean={np.nanmean(y[:, t]):.1f}, std={np.nanstd(y[:, t]):.1f}"
        )

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
    results, perm_imp_oof = nested_cv_regression(
        levels=levels,
        pool_maps_by_level=pool_maps,
        subject_ids=subject_ids,
        y=y,
        model_name=args.model,
        target_names=target_short_names,
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
        covariate_names=_COVARIATE_NAMES if covariates_all is not None else None,
        perm_imp_oof=args.perm_imp_oof,
        perm_imp_skip_node=args.skip_node_perm,
    )
    results_tag = f"{args.model}_{args.imputation}_{args.target_family}"
    results.to_csv(args.results_dir / f"regression_cv_results_{results_tag}.csv", index=False)

    # Write per-target OOF permutation importance.
    if perm_imp_oof:
        for granularity, target_to_gid_dict in perm_imp_oof.items():
            for tname, gid_dict in target_to_gid_dict.items():
                rows_pi = [
                    {"id": k, "importance_mean": v["mean"],
                     "importance_std": v["std"], "n_folds": v["n_folds"]}
                    for k, v in gid_dict.items()
                ]
                if not rows_pi:
                    continue
                df_pi = pd.DataFrame(rows_pi).sort_values("importance_mean", ascending=False)
                out = args.results_dir / (
                    f"perm_importance_{granularity}_{tname}_{args.model}_{args.target_family}.csv"
                )
                df_pi.to_csv(out, index=False)
        # Also a long-form "everything in one file" for easy downstream merging.
        long_rows = []
        for granularity, target_to_gid_dict in perm_imp_oof.items():
            for tname, gid_dict in target_to_gid_dict.items():
                for gid, vals in gid_dict.items():
                    long_rows.append({
                        "granularity": granularity, "target": tname, "id": gid,
                        "importance_mean": vals["mean"], "importance_std": vals["std"],
                        "n_folds": vals["n_folds"],
                    })
        if long_rows:
            pd.DataFrame(long_rows).to_csv(
                args.results_dir / f"perm_importance_long_{results_tag}.csv", index=False
            )
        print(f"  wrote per-target OOF perm-importance CSVs to {args.results_dir}")
    print("\nSummary (mean +/- std) per (method, target):")
    summary = results.groupby(["method", "target"]).agg(
        r_mean=("pearson_r", "mean"), r_std=("pearson_r", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n=("pearson_r", "count"),
    ).sort_index()
    print(summary.to_string())
    summary.to_csv(args.results_dir / f"regression_cv_summary_{results_tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
