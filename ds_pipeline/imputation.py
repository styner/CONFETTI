"""Per-subject imputation strategies.

Two interchangeable imputers:

  KnnImputer  -- fast (~ms/subject), per-subject k-Nearest-Neighbors in 3D
                  Euclidean space. NO cross-subject information is used, so no
                  fold leakage can occur. Default for iterating.

  SirenImputer -- per-subject SIREN implicit neural representation. Uses
                  cross-subject per-property mean/std for z-scoring; the
                  caller MUST pass training-fold subject IDs (`train_subjects`)
                  so the stats are computed without test-fold leakage. Default
                  hyperparameters per user spec: epochs=200, omega_0=10.

Both implementations are stand-alone (no import of impute_missing_profiles.py)
so the package is self-contained.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor

from .config import DEFAULT_SIREN_EPOCHS, DEFAULT_SIREN_OMEGA0, MISSING_VALUE, PROPERTIES


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


# Imputed property arrays follow the same nested-dict layout as
# LevelData.property_values: property -> subject_id -> 1D numpy array, but with
# all -1 sentinels replaced by predictions.
ImputedArrays = dict[str, dict[str, np.ndarray]]


def _observed_mask(arr: np.ndarray, missing_value: float = MISSING_VALUE) -> np.ndarray:
    return arr != missing_value


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


class KnnImputer:
    """Per-subject kNN imputation.

    For each subject:
      1. Identify points with observed values (mask using sentinel)
      2. Fit a k-NN regressor on (coords[obs] -> properties[obs])
      3. Predict at coords[missing]
    All properties are predicted from the same regressor in one call.

    Because each subject is imputed independently, the result does not depend
    on which subjects are in the training fold; the same imputed values can be
    re-used across folds (call `impute()` once before nested CV starts).
    """

    def __init__(self, k: int = 5, missing_value: float = MISSING_VALUE):
        self.k = k
        self.missing_value = missing_value

    def impute(
        self,
        coords: np.ndarray,
        property_values: dict[str, dict[str, np.ndarray]],
        properties: tuple[str, ...] = PROPERTIES,
        train_subjects: list[str] | None = None,  # ignored; here for API parity
    ) -> ImputedArrays:
        n = coords.shape[0]
        subjects = sorted(set().union(*(property_values[p].keys() for p in properties)))
        out: ImputedArrays = {p: {} for p in properties}

        for sid in subjects:
            # Build a (n, P) matrix of this subject's values; row j is missing
            # iff every property is -1 there (the input convention).
            P = len(properties)
            mat = np.full((n, P), self.missing_value, dtype=np.float32)
            for j, prop in enumerate(properties):
                if sid in property_values[prop]:
                    mat[:, j] = property_values[prop][sid]
            mask_obs = (mat != self.missing_value).all(axis=1)
            mask_miss = ~mask_obs
            if mask_miss.sum() == 0:
                for j, prop in enumerate(properties):
                    out[prop][sid] = mat[:, j].copy()
                continue
            if mask_obs.sum() < self.k:
                # Fall back to mean if too few observed points.
                fill = mat[mask_obs].mean(axis=0) if mask_obs.any() else np.zeros(P)
                mat[mask_miss] = fill
            else:
                knn = KNeighborsRegressor(n_neighbors=self.k, weights="distance")
                knn.fit(coords[mask_obs], mat[mask_obs])
                mat[mask_miss] = knn.predict(coords[mask_miss])
            for j, prop in enumerate(properties):
                out[prop][sid] = mat[:, j].copy()
        return out


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------


class _SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool, omega_0: float):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                bound = (6.0 / in_f) ** 0.5 / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class _Siren(nn.Module):
    def __init__(
        self,
        in_f: int,
        hidden_f: int,
        n_hidden: int,
        out_f: int,
        omega_0: float,
    ):
        super().__init__()
        layers: list[nn.Module] = [_SineLayer(in_f, hidden_f, True, omega_0)]
        for _ in range(n_hidden):
            layers.append(_SineLayer(hidden_f, hidden_f, False, omega_0))
        final = nn.Linear(hidden_f, out_f)
        with torch.no_grad():
            bound = (6.0 / hidden_f) ** 0.5 / omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class SirenImputer:
    """Per-subject SIREN imputation.

    Uses training-fold subjects only to compute the per-property mean/std for
    z-scoring (this is where leakage would happen if we used all subjects).
    Each subject's missing-value imputation depends on the subject's own
    observed (x, y, z) and the training-fold stats.
    """

    epochs: int = DEFAULT_SIREN_EPOCHS
    omega_0: float = DEFAULT_SIREN_OMEGA0
    hidden_features: int = 256
    hidden_layers: int = 3
    lr: float = 1e-4
    seed: int = 0
    device: str = "auto"
    missing_value: float = MISSING_VALUE

    def _select_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _per_property_stats(
        self,
        property_values: dict[str, dict[str, np.ndarray]],
        properties: tuple[str, ...],
        train_subjects: list[str],
    ) -> dict[str, tuple[float, float]]:
        train_set = set(train_subjects)
        stats: dict[str, tuple[float, float]] = {}
        for prop in properties:
            chunks = []
            for sid, arr in property_values[prop].items():
                if sid not in train_set:
                    continue
                mask = arr != self.missing_value
                if mask.any():
                    chunks.append(arr[mask])
            if not chunks:
                stats[prop] = (0.0, 1.0)
                continue
            cat = np.concatenate(chunks)
            mu = float(cat.mean())
            sd = float(cat.std()) or 1.0
            stats[prop] = (mu, sd)
        return stats

    def impute(
        self,
        coords: np.ndarray,
        property_values: dict[str, dict[str, np.ndarray]],
        properties: tuple[str, ...] = PROPERTIES,
        train_subjects: list[str] | None = None,
    ) -> ImputedArrays:
        if train_subjects is None:
            raise ValueError(
                "SirenImputer requires `train_subjects` to avoid leakage; pass "
                "the list of training-fold subject ids."
            )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._select_device()

        # Normalize coords to [-1, 1].
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        center = (lo + hi) / 2.0
        half = np.where(hi > lo, (hi - lo) / 2.0, 1.0)
        coords_n = ((coords - center) / half).astype(np.float32)
        coords_t = torch.tensor(coords_n, device=device)

        prop_stats = self._per_property_stats(property_values, properties, train_subjects)
        subjects = sorted(set().union(*(property_values[p].keys() for p in properties)))
        out: ImputedArrays = {p: {} for p in properties}
        n = coords.shape[0]
        P = len(properties)

        for s_idx, sid in enumerate(subjects):
            mat = np.full((n, P), self.missing_value, dtype=np.float32)
            for j, prop in enumerate(properties):
                if sid in property_values[prop]:
                    mat[:, j] = property_values[prop][sid]
            mask_obs = (mat != self.missing_value).all(axis=1)
            mask_miss = ~mask_obs
            if not mask_miss.any():
                for j, prop in enumerate(properties):
                    out[prop][sid] = mat[:, j].copy()
                continue
            if not mask_obs.any():
                # Fully missing: cannot impute; leave the sentinel.
                for j, prop in enumerate(properties):
                    out[prop][sid] = mat[:, j].copy()
                continue
            # z-score per property using training-fold stats.
            mat_z = mat.copy()
            for j, prop in enumerate(properties):
                mu, sd = prop_stats[prop]
                mat_z[mask_obs, j] = (mat[mask_obs, j] - mu) / sd
            torch.manual_seed(self.seed + s_idx)
            model = _Siren(
                in_f=3,
                hidden_f=self.hidden_features,
                n_hidden=self.hidden_layers,
                out_f=P,
                omega_0=self.omega_0,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            obs_t = coords_t[mask_obs]
            vals_t = torch.tensor(mat_z[mask_obs], dtype=torch.float32, device=device)
            for _ in range(self.epochs):
                pred = model(obs_t)
                loss = ((pred - vals_t) ** 2).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(coords_t[mask_miss]).cpu().numpy()
            for j, prop in enumerate(properties):
                mu, sd = prop_stats[prop]
                mat[mask_miss, j] = preds[:, j] * sd + mu
                out[prop][sid] = mat[:, j].copy()
        return out
