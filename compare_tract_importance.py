#!/usr/bin/env python3
"""Compare per-tract OOF permutation importance for LogReg vs the three GNNs.

Computes LogReg per-tract permutation importance under the EXACT same nested
5x3 stratified CV protocol as classify_ds.py uses for the GNNs, then loads the
already-saved GNN per-tract perm-importance CSVs and prints a head-to-head
comparison table.

Per fold:
  1. Get training subject IDs
  2. Compute training-fold covariate stats; z-score
  3. Build (S, P*T + n_cov) per-tract-mean + covariate feature matrix
     (P=5 properties, T=66 tracts, n_cov=3)
  4. Fit LogReg (balanced class weight, L2, liblinear) on the training fold
  5. For each tract: permute its P-column block in the *test* fold across
     subjects, recompute test-AUC, store the drop
  6. Accumulate per-tract drop values across all folds

Then aggregate (mean, std across 15 folds) and emit:
    ds_results/logreg_perm_importance/perm_importance_tract_logreg.csv
    ds_results/logreg_perm_importance/perm_importance_covariates_logreg.csv
    ds_results/logreg_perm_importance/comparison_top_tracts.csv  (LogReg vs each GNN)

Imputation: kNN per subject, computed once (no fold leakage). Matches
classify_ds.py exactly so the comparison is methodologically apples-to-apples.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ds_pipeline.baselines import per_tract_mean_features
from ds_pipeline.config import (
    COVARIATE_NAMES,
    DEFAULT_LEVEL_BASE,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTER_REPEATS,
    DEFAULT_SEED,
    PROPERTIES,
)
from ds_pipeline.cv import _apply_covariate_stats, _covariate_train_stats
from ds_pipeline.data import (
    build_covariate_matrix,
    load_levels,
    load_v06_subjects,
    select_subjects,
)
from ds_pipeline.imputation import KnnImputer


def load_tract_names(merged_csv: Path) -> dict[int, str]:
    """Read the merge_fibers.py output CSV mapping label -> filename, and turn
    each filename into a short tract name by stripping the standard suffix."""
    names: dict[int, str] = {}
    with merged_csv.open() as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            label = int(row[0])
            fname = row[1]
            tract = fname.replace("_parametrized_axis_profiles.vtk", "")
            tract = tract.replace("_parametrized_axis.vtk", "")
            tract = tract.replace(".vtk", "")
            names[label] = tract
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("OutputMerged"))
    p.add_argument("--level-base", default=DEFAULT_LEVEL_BASE)
    p.add_argument("--labels-csv", type=Path,
                   default=Path("Data_IBISEP_DS/EP_DS_DWI_data_V06.csv"))
    p.add_argument("--tract-name-csv", type=Path,
                   default=Path("OutputMerged/FiberAxisProfiles_merged.csv"))
    p.add_argument("--results-dir", type=Path,
                   default=Path("ds_results/logreg_perm_importance"))
    p.add_argument("--gnn-results-root", type=Path,
                   default=Path("ds_results/final_with_covariates"))
    p.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument("--repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-tract-repeats", type=int, default=3)
    p.add_argument("--n-cov-repeats", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10,
                   help="How many top tracts to show in the comparison table.")
    args = p.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Load data (L0 of the multi-resolution stack is all we need) ---
    print(f"Loading L0 from {args.data_dir} ...")
    levels = load_levels(args.data_dir, (0,), base=args.level_base, properties=PROPERTIES)
    ld0 = levels[0]
    print(f"  {ld0.coords.shape[0]} L0 points across {len(np.unique(ld0.fiber_label))} tracts")

    print(f"Loading labels from {args.labels_csv}")
    label_df = load_v06_subjects(args.labels_csv)
    subject_ids = select_subjects(levels, label_df, require_label=True)
    y = np.array([int(label_df.loc[sid, "label"]) for sid in subject_ids], dtype=np.int64)
    covariates = build_covariate_matrix(label_df, subject_ids)
    print(f"  {len(subject_ids)} subjects (DS={int((y==1).sum())}, Control={int((y==0).sum())})")
    print(f"  covariates shape={covariates.shape}, NaN={int(np.isnan(covariates).sum())}")

    # --- kNN-impute once (per-subject, no fold leakage) ---
    print("kNN imputation (cached, per-subject)...")
    knn_imputed = KnnImputer(k=5).impute(ld0.coords, ld0.property_values, PROPERTIES)

    # --- Build per-tract-mean feature matrix once ---
    print("Building per-tract-mean feature matrix ...")
    X_tract, col_names = per_tract_mean_features(
        ld0, subject_ids, knn_imputed, properties=PROPERTIES
    )
    print(f"  per-tract block shape={X_tract.shape}")
    P, T = len(PROPERTIES), len(np.unique(ld0.fiber_label))
    # Column layout from per_tract_mean_features: [P0 x T0..Tn-1, P1 x T0..Tn-1, ...]
    # So column for (property p, tract t) is p*T + t. We'll permute the
    # P-column block belonging to tract t = {p*T + t : p in 0..P-1}.

    # --- Stratified nested CV (5-fold, 3 repeats) -------------------------
    tract_drops: dict[int, list[float]] = defaultdict(list)
    cov_drops: dict[str, list[float]] = defaultdict(list)
    fold_aucs: list[float] = []

    n_subjects = len(subject_ids)
    n_cov = covariates.shape[1]
    print(
        f"Running nested CV: {args.outer_folds} outer x {args.repeats} repeats, "
        f"{args.n_tract_repeats} tract-perm repeats per fold ..."
    )

    fold_count = 0
    for rep in range(args.repeats):
        skf = StratifiedKFold(
            n_splits=args.outer_folds, shuffle=True, random_state=args.seed + rep
        )
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(subject_ids, y)):
            fold_count += 1
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Build full feature matrix for this fold: per-tract block + z-scored covariates.
            c_mu, c_sd = _covariate_train_stats(covariates, tr_idx)
            cov_full = _apply_covariate_stats(covariates, c_mu, c_sd)
            X_full = np.concatenate([X_tract, cov_full], axis=1)
            X_tr, X_te = X_full[tr_idx], X_full[te_idx]

            scaler = StandardScaler().fit(X_tr)
            Xtr_s = scaler.transform(X_tr)
            Xte_s = scaler.transform(X_te)
            clf = LogisticRegression(
                C=1.0, max_iter=5000, class_weight="balanced",
                solver="liblinear", random_state=args.seed,
            )
            clf.fit(Xtr_s, y_tr)
            base_proba = clf.predict_proba(Xte_s)[:, 1]
            if len(np.unique(y_te)) < 2:
                continue
            base_auc = float(roc_auc_score(y_te, base_proba))
            fold_aucs.append(base_auc)

            # Per-tract permutation: permute the P-column block for tract t in X_te
            for t in range(T):
                cols = np.array([p * T + t for p in range(P)])
                drops = []
                for r in range(args.n_tract_repeats):
                    perm = rng.permutation(Xte_s.shape[0])
                    Xte_perm = Xte_s.copy()
                    Xte_perm[:, cols] = Xte_s[perm][:, cols]
                    proba = clf.predict_proba(Xte_perm)[:, 1]
                    drops.append(base_auc - float(roc_auc_score(y_te, proba)))
                tract_drops[t].append(float(np.mean(drops)))

            # Per-covariate permutation: permute the n_cov tail columns in X_te
            cov_start = P * T
            for c_idx, name in enumerate(COVARIATE_NAMES):
                col = cov_start + c_idx
                drops = []
                for r in range(args.n_cov_repeats):
                    perm = rng.permutation(Xte_s.shape[0])
                    Xte_perm = Xte_s.copy()
                    Xte_perm[:, col] = Xte_s[perm, col]
                    proba = clf.predict_proba(Xte_perm)[:, 1]
                    drops.append(base_auc - float(roc_auc_score(y_te, proba)))
                cov_drops[name].append(float(np.mean(drops)))

    print(f"  done. CV AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # --- Aggregate ---
    tract_rows = []
    for t in range(T):
        vals = tract_drops.get(t, [])
        if not vals:
            continue
        tract_rows.append({
            "tract": t,
            "importance_mean": float(np.mean(vals)),
            "importance_std": float(np.std(vals)),
            "n_folds": len(vals),
        })
    tract_df = pd.DataFrame(tract_rows).sort_values("importance_mean", ascending=False)
    tract_df.to_csv(args.results_dir / "perm_importance_tract_logreg.csv", index=False)

    cov_rows = [
        {"covariate": name, "importance_mean": float(np.mean(vals)),
         "importance_std": float(np.std(vals)), "n_folds": len(vals)}
        for name, vals in cov_drops.items()
    ]
    cov_df = pd.DataFrame(cov_rows).sort_values("importance_mean", ascending=False)
    cov_df.to_csv(args.results_dir / "perm_importance_covariates_logreg.csv", index=False)

    print("\nLogReg covariate OOF perm-importance:")
    for r in cov_rows:
        print(f"  {r['covariate']:>22}: {r['importance_mean']:+.4f} ± {r['importance_std']:.4f}")

    # --- Load GNN per-tract files for comparison ---
    tract_names = load_tract_names(args.tract_name_csv)

    def load_gnn(model: str) -> pd.DataFrame | None:
        f = args.gnn_results_root / model / f"perm_importance_tract_{model}.csv"
        if not f.is_file():
            print(f"  warning: no GNN file at {f}")
            return None
        return pd.read_csv(f)

    gnn_dfs: dict[str, pd.DataFrame] = {}
    for m in ("gcn", "multi", "unet"):
        df = load_gnn(m)
        if df is not None:
            gnn_dfs[m] = df

    K = args.top_k
    methods = ["logreg", *gnn_dfs.keys()]
    top_tables: dict[str, list[tuple[int, float]]] = {}
    top_tables["logreg"] = [
        (int(row.tract), float(row.importance_mean))
        for row in tract_df.head(K).itertuples()
    ]
    for m, df in gnn_dfs.items():
        top_tables[m] = [
            (int(row.tract), float(row.importance_mean))
            for row in df.head(K).itertuples()
        ]

    print(f"\nTop-{K} most-predictive tracts per method (tract_id : importance, name):")
    print("=" * 78)
    for m in methods:
        print(f"\n[{m}]")
        for tid, imp in top_tables[m]:
            print(f"  tract {tid:>3}: {imp:+.4f}   {tract_names.get(tid, '?')}")

    # Build a side-by-side comparison CSV: union of top-K tracts across methods,
    # showing each method's importance value and rank (1-based, NaN if outside top-K).
    union_tracts: list[int] = []
    seen: set[int] = set()
    for m in methods:
        for tid, _ in top_tables[m]:
            if tid not in seen:
                seen.add(tid)
                union_tracts.append(tid)

    cmp_rows = []
    for tid in union_tracts:
        row = {"tract": tid, "name": tract_names.get(tid, "")}
        for m in methods:
            ranking = [r[0] for r in top_tables[m]]
            if tid in ranking:
                row[f"{m}_rank"] = ranking.index(tid) + 1
            else:
                row[f"{m}_rank"] = np.nan
            # Pull the actual importance value (across the full ranking)
            src = tract_df if m == "logreg" else gnn_dfs[m]
            row[f"{m}_imp"] = float(src.loc[src.tract == tid, "importance_mean"].iloc[0])
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(args.results_dir / "comparison_top_tracts.csv", index=False)

    # Rank correlations (Spearman) between methods over full 66-tract ranking.
    from scipy.stats import spearmanr
    method_rank_vecs: dict[str, np.ndarray] = {}
    for m in methods:
        src = tract_df if m == "logreg" else gnn_dfs[m]
        rank_by_tid = {int(r.tract): r.Index for r in src.sort_values("importance_mean", ascending=False).reset_index().itertuples()}
        method_rank_vecs[m] = np.array([rank_by_tid[t] for t in range(T)], dtype=np.float32)

    print("\nSpearman rank correlation between methods (over all 66 tracts):")
    print("=" * 78)
    print("        " + "  ".join(f"{m:>10}" for m in methods))
    for m1 in methods:
        row = [f"{m1:<8}"]
        for m2 in methods:
            rho, _ = spearmanr(method_rank_vecs[m1], method_rank_vecs[m2])
            row.append(f"{rho:>10.3f}")
        print("  ".join(row))

    print(f"\nSaved: {args.results_dir / 'perm_importance_tract_logreg.csv'}")
    print(f"Saved: {args.results_dir / 'perm_importance_covariates_logreg.csv'}")
    print(f"Saved: {args.results_dir / 'comparison_top_tracts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
