#!/usr/bin/env python3
"""Experiment 1: raw DS-vs-Control statistics at AC_olfactory (tract 0).

For every node belonging to tract 0 and every property in (AD, FA, NDI, ODI, RD):
  * two-sample t-test (Welch, unequal variances) DS vs Control
  * Cohen's d effect size
  * Benjamini-Hochberg FDR-corrected q-value across all (node, property) pairs

Inputs:
  * imputed merged VTK with per-(property, subject) point-data arrays
  * V06 label CSV (cohort -> DS/Control)
Outputs:
  * ds_results/validation_AC_olfactory/raw_ttest_AC_olfactory.csv
  * a small textual summary

The goal: show whether tract 0 has a *direct* DS/Control DTI difference at any
node, independent of any GNN. This validates whether the GNN finding rests on
real biological signal (significant t-tests) or on something the GNN constructs
via spatial coupling (no/few significant t-tests).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ds_pipeline.config import DEFAULT_LEVEL_BASE, PROPERTIES
from ds_pipeline.data import (
    load_levels,
    load_v06_subjects,
    select_subjects,
)
from ds_pipeline.imputation import KnnImputer


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges-style pooled-std Cohen's d (positive => a > b)."""
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s)


def main() -> int:
    data_dir = Path("OutputMerged")
    out_dir = Path("ds_results/validation_AC_olfactory")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading L0 ...")
    levels = load_levels(data_dir, (0,), base=DEFAULT_LEVEL_BASE, properties=PROPERTIES)
    ld0 = levels[0]

    print("Loading labels ...")
    label_df = load_v06_subjects(Path("Data_IBISEP_DS/EP_DS_DWI_data_V06.csv"))
    subject_ids = select_subjects(levels, label_df, require_label=True)
    y = np.array([int(label_df.loc[sid, "label"]) for sid in subject_ids], dtype=np.int64)
    n_ds = int((y == 1).sum()); n_co = int((y == 0).sum())
    print(f"  {len(subject_ids)} subjects (DS={n_ds}, Control={n_co})")

    # kNN-impute once so the per-node analysis isn't broken by per-subject NaNs.
    # (Same convention as the GNN runs; per-subject so no cross-subject leakage.)
    print("kNN-imputing missing per-(property, subject) values ...")
    imputed = KnnImputer(k=5).impute(ld0.coords, ld0.property_values, PROPERTIES)

    # Tract-0 node indices in the L0 frame.
    tract_zero_mask = (ld0.fiber_label == 0)
    tract_zero_nodes = np.where(tract_zero_mask)[0]
    print(f"AC_olfactory at L0: {tract_zero_nodes.size} nodes "
          f"(arclength range "
          f"[{ld0.arclength[tract_zero_nodes].min():.1f}, "
          f"{ld0.arclength[tract_zero_nodes].max():.1f}] mm)")

    rows = []
    for prop in PROPERTIES:
        subj_to_arr = imputed[prop]
        # Build (S, n_tract0_nodes) matrix
        S = len(subject_ids)
        mat = np.full((S, tract_zero_nodes.size), np.nan, dtype=np.float32)
        for s_idx, sid in enumerate(subject_ids):
            if sid in subj_to_arr:
                mat[s_idx] = subj_to_arr[sid][tract_zero_nodes]
        ds_rows = mat[y == 1]
        co_rows = mat[y == 0]
        for n_idx, node in enumerate(tract_zero_nodes):
            ds_vals = ds_rows[:, n_idx]
            co_vals = co_rows[:, n_idx]
            ds_vals = ds_vals[~np.isnan(ds_vals)]
            co_vals = co_vals[~np.isnan(co_vals)]
            if ds_vals.size < 5 or co_vals.size < 5:
                t_stat = np.nan; p_val = np.nan; d = np.nan
            else:
                t_stat, p_val = stats.ttest_ind(ds_vals, co_vals, equal_var=False)
                d = cohens_d(ds_vals, co_vals)
            rows.append({
                "node": int(node),
                "arclength_mm": float(ld0.arclength[node]),
                "property": prop,
                "n_DS": int(ds_vals.size),
                "n_Control": int(co_vals.size),
                "mean_DS": float(ds_vals.mean()) if ds_vals.size else np.nan,
                "mean_Control": float(co_vals.mean()) if co_vals.size else np.nan,
                "cohens_d": float(d),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            })

    df = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR across the joint (node, property) family.
    valid = df["p_value"].notna()
    df["q_value"] = np.nan
    if valid.sum() > 0:
        _, q_vals, _, _ = multipletests(
            df.loc[valid, "p_value"].values, method="fdr_bh"
        )
        df.loc[valid, "q_value"] = q_vals
    df["sig_q05"] = df["q_value"] < 0.05
    df.sort_values(["property", "node"], inplace=True)
    csv_path = out_dir / "raw_ttest_AC_olfactory.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print("\n" + "=" * 78)
    print("AC_olfactory raw DS-vs-Control statistics summary")
    print("=" * 78)
    n_tests = int(df["p_value"].notna().sum())
    print(f"Total tests: {n_tests} ({tract_zero_nodes.size} nodes x {len(PROPERTIES)} properties)")
    n_p05 = int((df["p_value"] < 0.05).sum())
    n_q05 = int(df["sig_q05"].sum())
    print(f"  raw p < 0.05:           {n_p05} ({100 * n_p05 / n_tests:.1f}%)")
    print(f"  FDR q < 0.05:           {n_q05} ({100 * n_q05 / n_tests:.1f}%)")
    print()
    print("Per-property counts at q < 0.05 (and mean |Cohen's d|):")
    for prop in PROPERTIES:
        sub = df[df.property == prop]
        n_sig = int(sub["sig_q05"].sum())
        d_mean = float(sub["cohens_d"].abs().mean())
        d_max = float(sub["cohens_d"].abs().max())
        # Median direction at significant nodes (positive => DS > Control)
        med_d = (
            float(sub.loc[sub.sig_q05, "cohens_d"].median())
            if n_sig else float("nan")
        )
        print(f"  {prop:>4}: {n_sig:>3}/{tract_zero_nodes.size} sig   "
              f"|d|_mean={d_mean:.2f}  |d|_max={d_max:.2f}  "
              f"median d(at sig nodes)={med_d:+.2f} "
              f"(positive=DS>Control)")
    print()
    print(f"Saved per-(node, property) detail: {csv_path}")
    print()

    # Property-wide effect summary: pooled t-test per property collapsing all
    # tract-0 nodes (mean per subject). This is the "tract-mean baseline view".
    print("Tract-mean comparison (collapse AC_olfactory into a single mean per subject):")
    for prop in PROPERTIES:
        subj_means = np.zeros(len(subject_ids), dtype=np.float64)
        for s_idx, sid in enumerate(subject_ids):
            arr = imputed[prop].get(sid)
            if arr is not None:
                subj_means[s_idx] = float(arr[tract_zero_nodes].mean())
            else:
                subj_means[s_idx] = np.nan
        ds = subj_means[y == 1]; co = subj_means[y == 0]
        ds = ds[~np.isnan(ds)]; co = co[~np.isnan(co)]
        if ds.size > 1 and co.size > 1:
            t, p = stats.ttest_ind(ds, co, equal_var=False)
            d = cohens_d(ds, co)
            print(f"  {prop:>4}: tract-mean   d={d:+.3f}  t={t:+.2f}  p={p:.2e}  "
                  f"DS={ds.mean():.4g}  Ctrl={co.mean():.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
