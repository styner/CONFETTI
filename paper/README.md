# Off-Grid Workshop submission

Anonymized submission targeting the MICCAI 2026 OFF-Grid workshop, written
against the official LNCS MICCAI template (`llncs.cls`, `splncs04.bst`).

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | The manuscript. ~8 pages content + 1 page references. |
| `llncs.cls` | Springer LNCS class file (copied from `template/`). |
| `splncs04.bst` | LNCS bibliography style (copied from `template/`). |
| `figures/fig_attr_ranking.pdf` | Cross-method tract attribution ranking (Fig 1). |
| `figures/fig_acolf_validation.pdf` | Raw effect + ablation panels (Fig 2). |
| `figures/fig_target_heatmap.pdf` | Per-V24-target tract attribution heatmap (Fig 3). |
| `figures/fig_pipeline.pdf` | Pipeline schematic (kept for reference; not currently embedded). |
| `figures/make_figs.py` | Regenerates all 4 figures from the result CSVs in `ds_results/`. |
| `template/` | Original MICCAI template, kept untouched for reference. |

## Build

Standard LNCS workflow. On a system with TeX Live or MacTeX installed:

```sh
cd paper
pdflatex main.tex
pdflatex main.tex   # run twice so cross-references resolve
```

For Overleaf, upload `main.tex`, `llncs.cls`, `splncs04.bst`, and the
`figures/` directory; set `main.tex` as the main document.

## Anonymization

* The `\author{Anonymized Authors}` and `\institute{Anonymized Affiliations}`
  blocks follow MICCAI's recommended anonymous form.
* No acknowledgments or disclosure-of-interest sections in the manuscript;
  add them at camera-ready time.
* The dataset is referred to only as "a pediatric DS cohort" without
  identifying the study, site, or collection details. Adjust at
  camera-ready time if a public dataset name is to be cited.

## Reproducing the numerical results

All numbers in the tables and figures come from CSVs already written under
`ds_results/` by the project's classification and regression scripts:

* `ds_results/final_with_covariates/<model>/` – DS classification with OOF
  permutation importance.
* `ds_results/v06_to_v24_regression_perm/<model>/` – V06→V24 prospective
  regression with per-target OOF permutation importance.
* `ds_results/validation_AC_olfactory/` – the 5 validation experiments.
* `ds_results/{logreg,xgboost}_perm_importance/` – per-tract attribution for
  the baselines.

`figures/make_figs.py` reads these CSVs directly; rerun it after any
result update to refresh the embedded figures.
