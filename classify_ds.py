#!/usr/bin/env python3
"""DS-vs-Control classification: nested CV + interpretability.

Pipeline:
  1. Load multi-resolution level VTKs + adjacency matrices.
  2. Load V06 cohort labels; merge "Control" + "Control DS Infant" -> class 0.
  3. Run nested 5x5 CV (repeated 3x) with baselines (LogReg + XGBoost on
     per-tract means) and one GNN architecture per --model:
       - gcn:   single-level GCN at the coarsest resolution
       - multi: multi-scale concatenation across all levels
       - unet:  hierarchical Graph U-Net with skip connections
  4. After CV, refit on all data and produce three spatial-attribution outputs
     written back as VTK point-data arrays for ParaView:
       - permutation importance per tract  -> 'PermImp_tract'
       - permutation importance per node   -> 'PermImp_node'
       - integrated gradients (subject-mean magnitude per node) -> 'IG_mean'
     A sparse GNNExplainer mask is also computed for one positive example.

CLI quickstart:
  python classify_ds.py
  python classify_ds.py --model unet --imputation siren --repeats 1 --outer-folds 5
  python classify_ds.py --quick   # fast smoke test
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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ds_pipeline.baselines import (
    LogRegClassifier,
    XGBoostClassifier,
    per_tract_mean_features,
)
from ds_pipeline.config import (
    COVARIATE_NAMES,
    DEFAULT_INNER_FOLDS,
    DEFAULT_LEVEL_BASE,
    DEFAULT_LEVELS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTER_REPEATS,
    DEFAULT_SEED,
    DEFAULT_SIREN_EPOCHS,
    DEFAULT_SIREN_OMEGA0,
    PROPERTIES,
)
from ds_pipeline.cv import (
    TrainConfig,
    nested_cv_classification,
)
from ds_pipeline.data import (
    build_covariate_matrix,
    load_levels,
    load_v06_subjects,
    pool_maps_from_levels,
    select_subjects,
)
from ds_pipeline.interpret import (
    gnn_explainer_sparse,
    integrated_gradients,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."),
                   help="Directory containing the level VTKs + adjacencies.")
    p.add_argument("--level-base", default=DEFAULT_LEVEL_BASE,
                   help="Stem used by build_neighborhood_graph (default: %(default)s).")
    p.add_argument("--labels-csv", type=Path, default=Path("Data_IBISEP_DS/EP_DS_DWI_data_V06.csv"))
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--model", choices=("gcn", "multi", "unet"), default="unet")
    p.add_argument("--imputation", choices=("knn", "siren"), default="knn")
    p.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    p.add_argument("--repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=400, help="GNN training epochs per fold.")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false",
                   help="Disable BatchNorm1d after each GCNConv (default: on).")
    p.set_defaults(use_batchnorm=True)
    p.add_argument("--no-curves", dest="save_curves", action="store_false",
                   help="Skip saving per-fold + summary training-curve PNGs.")
    p.set_defaults(save_curves=True)
    p.add_argument("--curves-dir", type=Path, default=None,
                   help="Directory for training-curve PNGs (default: <results-dir>/curves).")
    p.add_argument("--no-covariates", dest="use_covariates", action="store_false",
                   help="Skip concatenating subject covariates (sex, gestational age, num_DWI_artifact) at the model head and in baselines.")
    p.set_defaults(use_covariates=True)
    p.add_argument("--skip-node-perm", action="store_true",
                   help="Skip the per-node OOF permutation importance pass (slowest interpret step).")
    p.add_argument("--ablate-tracts", type=int, nargs="+", default=None,
                   help="Tract ids (FiberLabel values at L0) to ablate: their per-node "
                        "property channels are replaced by the training-fold property mean "
                        "(so the spatial input no longer carries any DS signal at those tracts, "
                        "while the graph structure stays intact). Useful for measuring how "
                        "much a specific tract contributes to predictions.")
    p.add_argument("--keep-tracts", type=int, nargs="+", default=None,
                   help="Inverse of --ablate-tracts: keep ONLY the named tract(s); ablate "
                        "everything else. Use a single tract id to test 'this tract alone'.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--results-dir", type=Path, default=Path("ds_results"))
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 1 repeat, 2 outer folds, 30 epochs.")
    p.add_argument("--skip-interpret", action="store_true")
    p.add_argument("--no-explain-oof", dest="explain_oof", action="store_false",
                   help="Disable per-DS-subject out-of-fold GNNExplainer (on by default).")
    p.set_defaults(explain_oof=True)
    p.add_argument("--explain-epochs", type=int, default=200,
                   help="Mask-optimization epochs per OOF GNNExplainer run (default: 200).")
    p.add_argument("--siren-epochs", type=int, default=DEFAULT_SIREN_EPOCHS)
    p.add_argument("--siren-omega0", type=float, default=DEFAULT_SIREN_OMEGA0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.repeats = 1
        args.outer_folds = 2
        args.inner_folds = 2
        args.epochs = 30

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load multi-resolution graph data ---
    print(f"Loading levels {args.levels} from {args.data_dir} ...")
    levels = load_levels(
        args.data_dir, tuple(args.levels), base=args.level_base, properties=PROPERTIES
    )
    pool_maps = pool_maps_from_levels(levels)
    for L, ld in levels.items():
        print(f"  L{L}: {ld.coords.shape[0]} nodes, {ld.adj.nnz // 2} undirected edges")

    # --- Load and merge labels ---
    print(f"Loading labels from {args.labels_csv}")
    df = load_v06_subjects(args.labels_csv)
    subject_ids = select_subjects(levels, df, require_label=True)
    if not subject_ids:
        raise SystemExit("No subjects with both VTK data and a label.")
    y = np.array([int(df.loc[sid, "label"]) for sid in subject_ids], dtype=np.int64)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    print(f"  classification subjects: {len(subject_ids)} (DS={n_pos}, Control={n_neg})")

    covariates_all = None
    if args.use_covariates:
        covariates_all = build_covariate_matrix(df, subject_ids)
        n_nan = int(np.isnan(covariates_all).sum())
        print(
            f"  covariates: {covariates_all.shape[1]} columns "
            f"({', '.join(COVARIATE_NAMES)}); {n_nan} NaN cells (mean-imputed per training fold)"
        )

    # --- Build baseline plug-ins ---
    L0 = min(levels)
    ld0 = levels[L0]

    def baseline_features(imputed, ids):
        X, _ = per_tract_mean_features(ld0, ids, imputed, properties=PROPERTIES)
        return X

    baseline_fns = {
        "logreg": LogRegClassifier(C=1.0, seed=args.seed).fit_predict,
        "xgboost": XGBoostClassifier(seed=args.seed).fit_predict,
    }

    # --- Nested CV ---
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_batchnorm=args.use_batchnorm,
        device=args.device,
    )
    print(f"\nNested CV: model={args.model}, imputation={args.imputation}, "
          f"outer={args.outer_folds}, inner={args.inner_folds}, repeats={args.repeats}")
    curves_dir = (args.curves_dir or (args.results_dir / "curves")) if args.save_curves else None
    if curves_dir is not None:
        curves_dir.mkdir(parents=True, exist_ok=True)

    # Resolve --ablate-tracts / --keep-tracts into a single ablate_tracts set.
    ablate_set: set[int] | None = None
    if args.ablate_tracts is not None and args.keep_tracts is not None:
        raise SystemExit("Pass either --ablate-tracts or --keep-tracts, not both.")
    if args.ablate_tracts is not None:
        ablate_set = {int(t) for t in args.ablate_tracts}
        print(f"  ablating tracts: {sorted(ablate_set)}")
    elif args.keep_tracts is not None:
        keep_set = {int(t) for t in args.keep_tracts}
        all_tracts = set(int(t) for t in np.unique(levels[min(levels)].fiber_label))
        ablate_set = all_tracts - keep_set
        print(f"  keeping ONLY tracts {sorted(keep_set)}; ablating {len(ablate_set)} others")
    results, oof_explanations, perm_imp_oof = nested_cv_classification(
        levels=levels,
        pool_maps_by_level=pool_maps,
        subject_ids=subject_ids,
        y=y,
        model_name=args.model,
        baseline_fns=baseline_fns,
        baseline_feature_extractor=baseline_features,
        imputation=args.imputation,
        siren_cfg={"epochs": args.siren_epochs, "omega_0": args.siren_omega0, "seed": args.seed},
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        repeats=args.repeats,
        seed=args.seed,
        train_cfg=train_cfg,
        properties=PROPERTIES,
        n_outputs=1,
        explain_ds_oof=args.explain_oof,
        explain_epochs=args.explain_epochs,
        curves_dir=curves_dir,
        covariates=covariates_all,
        covariate_names=COVARIATE_NAMES if covariates_all is not None else None,
        perm_imp_oof=(not args.skip_interpret),
        perm_imp_skip_node=args.skip_node_perm,
        perm_imp_tract_repeats=3,
        perm_imp_node_repeats=2,
        perm_imp_cov_repeats=5,
        ablate_tracts=ablate_set,
    )
    results_csv = args.results_dir / f"ds_cv_results_{args.model}_{args.imputation}.csv"
    results.to_csv(results_csv, index=False)
    print(f"\nSaved per-fold results to {results_csv}")
    print("\nSummary (mean +/- std):")
    summary = results.groupby("method").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        n=("auc", "count"),
    ).sort_values("auc_mean", ascending=False)
    print(summary.to_string())
    summary.to_csv(args.results_dir / f"ds_cv_summary_{args.model}_{args.imputation}.csv")

    # --- OOF permutation importance CSVs ---
    if perm_imp_oof.get("tract"):
        rows_t = [
            {"tract": k, "importance_mean": v["mean"], "importance_std": v["std"],
             "n_folds": v["n_folds"]}
            for k, v in sorted(perm_imp_oof["tract"].items())
        ]
        pd.DataFrame(rows_t).sort_values("importance_mean", ascending=False).to_csv(
            args.results_dir / f"perm_importance_tract_{args.model}.csv", index=False,
        )
    if perm_imp_oof.get("covariate"):
        rows_c = [
            {"covariate": k, "importance_mean": v["mean"], "importance_std": v["std"],
             "n_folds": v["n_folds"]}
            for k, v in perm_imp_oof["covariate"].items()
        ]
        pd.DataFrame(rows_c).sort_values("importance_mean", ascending=False).to_csv(
            args.results_dir / f"perm_importance_covariates_{args.model}.csv", index=False,
        )
        print("OOF covariate permutation importance:")
        for row in rows_c:
            print(f"  {row['covariate']:>22}: {row['importance_mean']:+.4f} ± {row['importance_std']:.4f}")
    if perm_imp_oof.get("node"):
        rows_n = [
            {"node": k, "importance_mean": v["mean"], "importance_std": v["std"],
             "n_folds": v["n_folds"]}
            for k, v in sorted(perm_imp_oof["node"].items())
        ]
        pd.DataFrame(rows_n).sort_values("importance_mean", ascending=False).to_csv(
            args.results_dir / f"perm_importance_node_{args.model}.csv", index=False,
        )

    # --- OOF GNNExplainer outputs (per DS subject) ---
    oof_node_mean = None
    if oof_explanations:
        rows_long: list[dict] = []
        feat_rows: list[dict] = []
        L_used = next(iter(oof_explanations.values()))[0]["level"]
        per_subject_node_avg: dict[str, np.ndarray] = {}
        feat_channels = list(PROPERTIES) + ["x", "y", "z", "arclength"]
        for sid, entries in oof_explanations.items():
            # If the subject was OOF in multiple repeats, average across them.
            nm = np.mean([e["node_mask"] for e in entries], axis=0)
            fm = np.mean([e["feature_mask"] for e in entries], axis=0)
            avg_score = float(np.mean([e["score"] for e in entries]))
            per_subject_node_avg[sid] = nm
            for node_idx, v in enumerate(nm):
                rows_long.append({
                    "subject_id": sid, "node": int(node_idx), "mask": float(v),
                    "score": avg_score, "level": L_used,
                })
            feat_rows.append({
                "subject_id": sid, "score": avg_score, "level": L_used,
                **{f"mask_{ch}": float(fm[i]) for i, ch in enumerate(feat_channels)},
            })
        pd.DataFrame(rows_long).to_csv(
            args.results_dir / f"oof_gnnexplainer_nodes_{args.model}.csv",
            index=False,
        )
        pd.DataFrame(feat_rows).to_csv(
            args.results_dir / f"oof_gnnexplainer_features_{args.model}.csv",
            index=False,
        )
        oof_node_mean = np.mean(list(per_subject_node_avg.values()), axis=0).astype(np.float32)
        print(
            f"OOF GNNExplainer: {len(oof_explanations)} DS subjects explained "
            f"at level L{L_used}; per-subject masks saved."
        )

    if args.skip_interpret:
        # Even when --skip-interpret, write the OOF mean mask back to a VTK so
        # the per-subject explanations have a spatial summary.
        if oof_node_mean is not None:
            from ds_pipeline.config import level_vtk_path
            L_used = next(iter(oof_explanations.values()))[0]["level"]
            src_path = level_vtk_path(args.data_dir, L_used, args.level_base)
            reader = vtk.vtkPolyDataReader(); reader.SetFileName(str(src_path)); reader.Update()
            src_pd = reader.GetOutput()
            arr = numpy_to_vtk(oof_node_mean, deep=True); arr.SetName("OofGNNExplainer_DSmean")
            src_pd.GetPointData().AddArray(arr)
            src_pd.GetPointData().SetActiveScalars("OofGNNExplainer_DSmean")
            out_vtk = args.results_dir / f"ds_oof_gnnexplainer_{args.model}_L{L_used}.vtk"
            w = vtk.vtkPolyDataWriter(); w.SetFileName(str(out_vtk)); w.SetInputData(src_pd)
            w.SetFileTypeToBinary(); w.Write()
            print(f"  OOF mean mask -> {out_vtk}")
        return 0

    # --- Final refit + interpretability ---
    print("\nRefitting on all subjects for interpretability...")
    from ds_pipeline.cv import (
        GNNTrainer,
        _apply_property_stats,
        _build_inputs,
        _maybe_impute,
        _per_property_train_stats,
        _propagate_imputed_to_levels,
    )

    imputed_L0 = _maybe_impute(
        levels, subject_ids, args.imputation, None,
        {"epochs": args.siren_epochs, "omega_0": args.siren_omega0, "seed": args.seed},
        PROPERTIES,
    )
    imputed_per_level = _propagate_imputed_to_levels(levels, imputed_L0, pool_maps)

    # Covariates for the post-CV refit: z-score using ALL subjects (refit-all
    # convention; no held-out test). Same NaN-mean-impute rule.
    cov_all = None
    if covariates_all is not None:
        from ds_pipeline.cv import _covariate_train_stats, _apply_covariate_stats
        all_idx = np.arange(covariates_all.shape[0])
        c_mu, c_sd = _covariate_train_stats(covariates_all, all_idx)
        cov_all = _apply_covariate_stats(covariates_all, c_mu, c_sd)

    trainer = GNNTrainer(args.model, train_cfg, "classification")
    test_logits, val_metric, info = trainer.fit_predict(
        levels=levels,
        pool_maps_by_level=pool_maps,
        train_subject_ids=subject_ids,
        val_subject_ids=[],
        test_subject_ids=subject_ids[:1],  # one dummy test subject, ignored
        imputed_per_level=imputed_per_level,
        y_train=y,
        y_val=None,
        properties=PROPERTIES,
        n_outputs=1,
        covariates_train=cov_all,
        covariates_val=None,
        covariates_test=cov_all[:1] if cov_all is not None else None,
    )
    n_covariates = 0 if cov_all is None else int(cov_all.shape[1])

    # Rebuild model + load best state for interpretation calls.
    from ds_pipeline.models import build_model
    device = train_cfg.select_device()
    used_levels = [max(args.levels)] if args.model == "gcn" else sorted(args.levels)
    in_dim = len(PROPERTIES) + 4
    n_lv = 1 if args.model == "gcn" else len(used_levels)
    model = build_model(args.model, in_dim=in_dim, n_levels=n_lv,
                        out_dim=1, hidden_dim=args.hidden_dim, dropout=args.dropout,
                        use_batchnorm=args.use_batchnorm,
                        n_covariates=n_covariates).to(device)
    model.load_state_dict(info["model_state"])

    # Build feature tensors for the *test* subjects (all of them here).
    prop_stats = _per_property_train_stats(levels, subject_ids, imputed_per_level, PROPERTIES)
    feats_np, _, _ = _build_inputs(levels, subject_ids, imputed_per_level, PROPERTIES)
    _apply_property_stats(feats_np, prop_stats, PROPERTIES)
    feats_used = {L: feats_np[L] for L in used_levels}
    sorted_used = sorted(used_levels)
    pool_maps_list = [
        pool_maps[(L, Lp1)] for L, Lp1 in zip(sorted_used[:-1], sorted_used[1:])
    ] if args.model == "unet" else []

    # NOTE: tract/node/covariate permutation importance are computed OOF
    # during nested CV (see perm_imp_oof above) -- not on the refit-all
    # model, which would suffer the training-data overfit ceiling effect.
    # IG and the confident-subject GNNExplainer below are local sensitivity /
    # sparse-mask tools that don't have the same ceiling issue, so they stay
    # on the refit-all model.
    L_used = used_levels[0]

    # --- Integrated Gradients (averaged across positive subjects) ---
    print("Integrated gradients (positive-class subjects)...")
    pos_mask = (y == 1)
    feats_pos = {L: feats_np[L][pos_mask] for L in used_levels}
    cov_pos = cov_all[pos_mask] if cov_all is not None else None
    ig = integrated_gradients(
        model, feats_pos, levels, used_levels, pool_maps_list,
        target_idx=0, steps=24, device=device, covariates=cov_pos,
    )
    # Mean over subjects, magnitude over feature channels.
    ig_primary = np.abs(ig[L_used]).mean(axis=0).mean(axis=-1)

    # --- GNNExplainer on the most confident positive prediction ---
    print("GNNExplainer on the most confident DS prediction...")
    feats_all = {L: feats_np[L] for L in used_levels}
    with torch.no_grad():
        from ds_pipeline.interpret import _predict
        scores = _predict(model, feats_all, levels, used_levels, pool_maps_list,
                          is_unet=(args.model == "unet"), device=device,
                          covariates=cov_all)
    scores_flat = scores.squeeze(-1)
    pos_idx = int(np.argmax(scores_flat * (y == 1)))
    cov_one = cov_all[pos_idx : pos_idx + 1] if cov_all is not None else None
    node_mask, feat_mask = gnn_explainer_sparse(
        model, feats_all, pos_idx, levels, used_levels, pool_maps_list,
        epochs=200, device=device, covariates=cov_one,
    )
    pd.DataFrame({"node": np.arange(node_mask.size), "mask": node_mask}).to_csv(
        args.results_dir / f"gnnexplainer_nodes_{args.model}_subject{pos_idx}.csv",
        index=False,
    )
    pd.DataFrame({"channel": [*PROPERTIES, "x", "y", "z", "arclength"], "mask": feat_mask}).to_csv(
        args.results_dir / f"gnnexplainer_features_{args.model}_subject{pos_idx}.csv",
        index=False,
    )

    # --- Write spatial maps back to a VTK for ParaView ---
    print("Writing spatial maps to VTK...")
    from ds_pipeline.config import level_vtk_path
    src_path = level_vtk_path(args.data_dir, L_used, args.level_base)
    reader = vtk.vtkPolyDataReader(); reader.SetFileName(str(src_path)); reader.Update()
    src_pd = reader.GetOutput()
    fiber_label = levels[L_used].fiber_label
    n_nodes = fiber_label.shape[0]
    attached: list[tuple[str, np.ndarray]] = []
    # OOF-aggregated tract perm importance broadcast to nodes.
    tract_mean = perm_imp_oof.get("tract", {})
    if tract_mean:
        tract_node = np.array(
            [tract_mean.get(int(fiber_label[i]), {"mean": 0.0})["mean"] for i in range(n_nodes)],
            dtype=np.float32,
        )
        attached.append(("PermImp_tract_OOF", tract_node))
    node_mean = perm_imp_oof.get("node", {})
    if node_mean:
        node_imp = np.array(
            [node_mean.get(int(i), {"mean": 0.0})["mean"] for i in range(n_nodes)],
            dtype=np.float32,
        )
        attached.append(("PermImp_node_OOF", node_imp))
    attached.append(("IG_mean", ig_primary.astype(np.float32)))
    if oof_node_mean is not None and oof_node_mean.shape[0] == n_nodes:
        attached.append(("OofGNNExplainer_DSmean", oof_node_mean))
    for name, arr in attached:
        varr = numpy_to_vtk(arr, deep=True)
        varr.SetName(name)
        src_pd.GetPointData().AddArray(varr)
    if any(name == "PermImp_tract_OOF" for name, _ in attached):
        src_pd.GetPointData().SetActiveScalars("PermImp_tract_OOF")
    out_vtk = args.results_dir / f"ds_interpretation_{args.model}_L{L_used}.vtk"
    w = vtk.vtkPolyDataWriter(); w.SetFileName(str(out_vtk)); w.SetInputData(src_pd)
    w.SetFileTypeToBinary(); w.Write()
    print(f"  -> {out_vtk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
