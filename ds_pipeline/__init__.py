"""Multi-resolution graph-learning pipeline for DS classification and
Vineland multi-output regression on fiber-axis profile data.

Modules:
  config      -- constants (property list, defaults, file conventions)
  data        -- label parsing, VTK loading, per-subject tensor assembly
  imputation  -- per-subject kNN (fast, used while iterating) and per-fold
                 SIREN (slower, used for final published results)
  baselines   -- linear logistic regression + XGBoost on per-tract means
  models      -- SingleLevelGCN, MultiScaleConcat, HierarchicalGraphUNet
  cv          -- nested cross-validation harness
  interpret   -- permutation importance, integrated gradients, GNNExplainer
"""
