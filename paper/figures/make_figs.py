#!/usr/bin/env python3
"""Generate the figures used in the off-grid workshop paper.

All figures are saved as both PDF (for LaTeX inclusion) and PNG (for quick
viewing) in the paper/figures/ directory.

Figures produced:
  fig_attr_ranking.pdf   -- mean rank of the top tracts across 7 attribution
                            methods (3 GNNs + LogReg perm/coef + XGB perm/gain)
  fig_target_heatmap.pdf -- per-target tract attribution heatmap for the
                            V06->V24 prospective regression (unet)
  fig_acolf_validation.pdf -- 2-panel validation: per-property Cohen's d at
                            AC_olfactory + ablation AUC summary
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RES = ROOT / "ds_results"


def tract_name_map() -> dict[int, str]:
    names: dict[int, str] = {}
    with open(ROOT / "OutputMerged" / "FiberAxisProfiles_merged.csv") as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            names[int(row[0])] = (
                row[1]
                .replace("_parametrized_axis_profiles.vtk", "")
                .replace("_parametrized_axis.vtk", "")
            )
    return names


# ---------------------------------------------------------------------------
# Figure 1: Cross-method tract-attribution ranking (top-15 tracts)
# ---------------------------------------------------------------------------


def fig_attr_ranking():
    names = tract_name_map()
    methods = {
        "GCN":   ("final_with_covariates/gcn/perm_importance_tract_gcn.csv", "importance_mean"),
        "Multi": ("final_with_covariates/multi/perm_importance_tract_multi.csv", "importance_mean"),
        "U-Net": ("final_with_covariates/unet/perm_importance_tract_unet.csv", "importance_mean"),
        "LogReg ($|\\beta|$)": ("logreg_perm_importance/abs_coef_tract_logreg.csv", "sum_abs_coef"),
        "XGBoost (gain)": ("xgboost_perm_importance/gain_importance_tract_xgboost.csv", "gain_sum"),
        "XGBoost (perm)": ("xgboost_perm_importance/perm_importance_tract_xgboost.csv", "importance_mean"),
    }
    T = 66
    rank_vecs: dict[str, np.ndarray] = {}
    for m, (p, c) in methods.items():
        df = pd.read_csv(RES / p)
        col_id = "tract" if "tract" in df.columns else "id"
        d = dict(zip(df[col_id].astype(int), df[c]))
        v = np.array([d.get(i, 0.0) for i in range(T)])
        # rank: 1 = most important
        order = np.argsort(-v)
        rank = np.empty(T, dtype=np.int32)
        for r, t in enumerate(order):
            rank[t] = r + 1
        rank_vecs[m] = rank

    # Take top tracts by their mean rank across GNN methods (1=best).
    gnn_mean_rank = np.mean(np.stack([rank_vecs[m] for m in ("GCN", "Multi", "U-Net")]), axis=0)
    top_idx = np.argsort(gnn_mean_rank)[:12]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.13
    x = np.arange(len(top_idx))
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628", "#999999"]
    for i, m in enumerate(methods):
        ranks = rank_vecs[m][top_idx]
        # plot inverted: smaller rank = taller bar
        bars = ax.bar(x + (i - 2.5) * width, 67 - ranks, width, label=m, color=colors[i])
    ax.set_xticks(x)
    short = [names.get(int(t), str(t)).replace("_", " ")[:18] for t in top_idx]
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Inverse rank (66 = top-ranked, 1 = least)", fontsize=9)
    ax.set_title(
        "Tract attribution across attribution methods (top 12 tracts by mean GNN rank)",
        fontsize=10,
    )
    ax.legend(ncol=3, loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim(0, 70)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_attr_ranking.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_attr_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: V06->V24 per-target tract attribution heatmap (top tracts, unet)
# ---------------------------------------------------------------------------


def fig_target_heatmap():
    names = tract_name_map()
    targets = [
        "V24_ABC", "V24_Comm", "V24_DailyLiving", "V24_Motor", "V24_Soc",
        "V24_BayleyCOG", "V24_BayleyEC", "V24_BayleyRC",
    ]
    short_targets = ["ABC", "Comm", "DailyLiv", "Motor", "Soc", "BayleyCOG", "BayleyEC", "BayleyRC"]
    T = 66
    mat = np.zeros((T, len(targets)), dtype=float)
    for j, tgt in enumerate(targets):
        df = pd.read_csv(
            RES / f"v06_to_v24_regression_perm/unet/perm_importance_tract_{tgt}_unet_v24_all.csv"
        )
        d = dict(zip(df["id"].astype(int), df["importance_mean"]))
        for i in range(T):
            mat[i, j] = d.get(i, 0.0)
    # Show the top-12 tracts by sum-of-importance across targets
    score = mat.sum(axis=1)
    top_idx = np.argsort(-score)[:12]
    sub = mat[top_idx]
    labels = [names.get(int(t), str(t)).replace("_", " ")[:22] for t in top_idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(sub, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(short_targets, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=8)
    # Annotate the top cell per column with a marker
    for j in range(len(targets)):
        i_max = int(np.argmax(sub[:, j]))
        ax.text(j, i_max, "•", ha="center", va="center", color="white", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("MAE increase under tract permutation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(
        "Per-target tract attribution: V06 imaging $\\rightarrow$ V24 outcomes (U-Net)",
        fontsize=10,
    )
    ax.set_xlabel("V24 outcome target", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_target_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_target_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: AC_olfactory validation -- 2 panels (raw effect + ablation)
# ---------------------------------------------------------------------------


def fig_acolf_validation():
    # Panel A: raw |Cohen's d| at AC_olfactory nodes per property
    raw = pd.read_csv(RES / "validation_AC_olfactory" / "raw_ttest_AC_olfactory.csv")
    props = ["AD", "FA", "NDI", "ODI", "RD"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    for pi, prop in enumerate(props):
        sub = raw[raw["property"] == prop].sort_values("arclength_mm")
        ax.plot(sub["arclength_mm"], sub["cohens_d"], lw=1.4, label=prop)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axhline(0.5, color="gray", lw=0.4, ls=":")
    ax.axhline(-0.5, color="gray", lw=0.4, ls=":")
    ax.set_xlabel("Arclength along AC olfactory (mm)", fontsize=9)
    ax.set_ylabel("Cohen's d (DS - Control)", fontsize=9)
    ax.set_title("Raw group difference along the tract", fontsize=10)
    ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    # Panel B: ablation AUC summary
    ax = axes[1]
    configs = ["Full graph", "Ablate tract 0", "Only tract 0"]
    base = {"GCN": 0.913, "Multi": 0.931, "U-Net": 0.940}
    abl = {"GCN": 0.959, "Multi": 0.929, "U-Net": 0.944}
    only = {"GCN": 0.761, "Multi": 0.774, "U-Net": 0.752}
    width = 0.23
    x = np.arange(3)
    for i, m in enumerate(("GCN", "Multi", "U-Net")):
        vals = [base[m], abl[m], only[m]]
        ax.bar(x + (i - 1) * width, vals, width, label=m,
               color=["#377eb8", "#ff7f00", "#4daf4a"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("CV AUC", fontsize=9)
    ax.set_title("DS classification: necessity / sufficiency", fontsize=10)
    ax.axhline(0.984, color="gray", lw=0.6, ls="--", label="XGBoost")
    ax.legend(fontsize=7, loc="lower left", frameon=False, ncol=2)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "AC olfactory validation: raw biology (left) and ablation (right)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_acolf_validation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_acolf_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline schematic (simple, programmatic)
# ---------------------------------------------------------------------------


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.5, 0.12, "Per-tract\nfibers"),
        (0.18, 0.5, 0.12, "Fiber-axis\ncomputation"),
        (0.34, 0.5, 0.13, "Multi-res.\nneighborhood\ngraph (L0..L6)"),
        (0.51, 0.5, 0.10, "SIREN\nimputation"),
        (0.65, 0.5, 0.13, "Per-subject\nnode features\n(5 DTI + 4 spatial)"),
        (0.82, 0.5, 0.15, "Graph U-Net\n+ covariates\n$\\rightarrow$ DS / V24"),
    ]
    for x, y, w, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.15), w, 0.3, fill=True,
                                    facecolor="#dde6f0", edgecolor="#1f4e79"))
        ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=8)
    # arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i + 1][0]
        ax.annotate(
            "", xy=(x_end, 0.5), xytext=(x_start, 0.5),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#1f4e79"),
        )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    print("  wrote fig_pipeline")
    fig_attr_ranking()
    print("  wrote fig_attr_ranking")
    fig_target_heatmap()
    print("  wrote fig_target_heatmap")
    fig_acolf_validation()
    print("  wrote fig_acolf_validation")
