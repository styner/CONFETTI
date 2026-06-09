"""Non-graph baselines: linear logistic regression and XGBoost on per-tract
mean property values.

For each (subject, tract, property), we compute the mean of the property over
all nodes belonging to the tract on the finest resolution (L0). Stacking these
gives a flat (P * T)-dim feature vector per subject, where P = number of
properties and T = number of tracts. The same feature space is reused for
both classification (DS vs Control) and multi-output regression (Vineland).

Both baselines are deliberately simple and well-regularized. They are not
expected to be beaten by GNNs on the DS task -- the user reported near-perfect
linear classification in prior work -- but they establish a calibration point
for the harness and for the GNNs' interpretability claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

try:  # XGBoost is heavy; degrade gracefully if absent.
    from xgboost import XGBClassifier, XGBRegressor
    _HAVE_XGB = True
except Exception:  # pragma: no cover
    _HAVE_XGB = False

from .config import PROPERTIES
from .data import LevelData


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def per_tract_mean_features(
    level0: LevelData,
    subject_ids: list[str],
    imputed_values: dict[str, dict[str, np.ndarray]] | None = None,
    properties: tuple[str, ...] = PROPERTIES,
) -> tuple[np.ndarray, list[str]]:
    """Return (S, P*T) per-tract-mean feature matrix and the column names.

    Tracts are ordered by the unique sorted values in `level0.fiber_label`.
    Within-property block is contiguous: [prop1_tract1, prop1_tract2, ...,
    prop2_tract1, ...].
    """
    tract_ids = np.unique(level0.fiber_label)
    src = imputed_values if imputed_values is not None else level0.property_values
    # Precompute per-tract masks
    masks = {int(t): (level0.fiber_label == t) for t in tract_ids}
    n_subj, n_prop, n_tract = len(subject_ids), len(properties), len(tract_ids)
    X = np.zeros((n_subj, n_prop * n_tract), dtype=np.float32)
    col_names: list[str] = []
    for p_idx, prop in enumerate(properties):
        for t_idx, tract in enumerate(tract_ids):
            col_names.append(f"{prop}_tract{int(tract)}")
            mask = masks[int(tract)]
            col = p_idx * n_tract + t_idx
            for s_idx, sid in enumerate(subject_ids):
                arr = src[prop].get(sid)
                if arr is None:
                    X[s_idx, col] = 0.0
                else:
                    vals = arr[mask]
                    X[s_idx, col] = float(vals.mean()) if vals.size else 0.0
    return X, col_names


# ---------------------------------------------------------------------------
# Classification baselines
# ---------------------------------------------------------------------------


@dataclass
class LogRegClassifier:
    """L2-regularized logistic regression on standardized features."""

    C: float = 1.0
    max_iter: int = 5000
    seed: int = 0

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.seed,
        )
        clf.fit(Xtr, y_train)
        return clf.predict_proba(Xte)[:, 1]


@dataclass
class XGBoostClassifier:
    """Gradient-boosted trees on raw features (XGBoost handles scale)."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    seed: int = 0

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        if not _HAVE_XGB:
            raise RuntimeError("xgboost not installed")
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        spw = (n_neg / max(n_pos, 1)) if n_pos else 1.0
        clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            random_state=self.seed,
            eval_metric="logloss",
            tree_method="hist",
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# Regression baselines (multi-output Vineland)
# ---------------------------------------------------------------------------


@dataclass
class RidgeRegressor:
    """L2-regularized multi-output ridge regression on standardized features."""

    alpha: float = 1.0
    seed: int = 0

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        n_out = y_train.shape[1] if y_train.ndim > 1 else 1
        preds = np.zeros((X_test.shape[0], n_out), dtype=np.float32)
        for i in range(n_out):
            yi = y_train[:, i] if y_train.ndim > 1 else y_train
            mask = ~np.isnan(yi)
            if mask.sum() < 5:
                preds[:, i] = np.nanmean(yi) if mask.any() else 0.0
                continue
            reg = Ridge(alpha=self.alpha, random_state=self.seed)
            reg.fit(Xtr[mask], yi[mask])
            preds[:, i] = reg.predict(Xte)
        return preds


@dataclass
class XGBoostRegressor:
    """Gradient-boosted trees (multi-output via per-target models)."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    seed: int = 0

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        if not _HAVE_XGB:
            raise RuntimeError("xgboost not installed")
        n_out = y_train.shape[1] if y_train.ndim > 1 else 1
        preds = np.zeros((X_test.shape[0], n_out), dtype=np.float32)
        for i in range(n_out):
            yi = y_train[:, i] if y_train.ndim > 1 else y_train
            mask = ~np.isnan(yi)
            if mask.sum() < 5:
                preds[:, i] = np.nanmean(yi) if mask.any() else 0.0
                continue
            reg = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.seed,
                tree_method="hist",
            )
            reg.fit(X_train[mask], yi[mask])
            preds[:, i] = reg.predict(X_test)
        return preds
