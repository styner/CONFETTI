"""Constants and defaults for the DS / Vineland pipeline."""

from pathlib import Path

# Diffusion properties actually used (per user spec: drop MD for collinearity with
# AD/RD, drop FWF for CSF noise). Order is the canonical column order downstream.
PROPERTIES: tuple[str, ...] = ("AD", "FA", "NDI", "ODI", "RD")

# Per-node spatial features (always shared across subjects, never imputed).
SPATIAL_FEATURE_NAMES: tuple[str, ...] = ("x", "y", "z", "arclength")

# Combined input dimension per node.
NODE_FEATURE_DIM = len(PROPERTIES) + len(SPATIAL_FEATURE_NAMES)

# Sentinel value marking missing entries in the merged VTK.
MISSING_VALUE = -1.0

# V06 cohort label mapping. "Control" and "Control DS Infant" merge to a single
# control class (label 0); "DS Infant" is the positive class (label 1).
COHORT_COLUMN = "demographics,Cohort"
COHORT_TO_LABEL: dict[str, int] = {
    "Control": 0,
    "Control DS Infant": 0,
    "DS Infant": 1,
}

# Subject-level covariates concatenated at the head of every model (and at the
# end of the per-tract-mean baseline feature vector). Order is canonical and
# fixed: ('sex', 'gestational_age', 'num_DWI_artifact').
#   * sex: 'Female' -> 0.0, 'Male' -> 1.0
#   * gestational_age: float weeks (some V06 subjects missing -> training-fold
#     mean imputation)
#   * num_DWI_artifact: integer count (DWI quality)
# All three are z-scored using TRAINING-FOLD stats only (no leakage).
COVARIATE_CSV_COLUMNS: tuple[str, ...] = (
    "demographics,Sex",
    "gestational_age",
    "num_DWI_artifact",
)
COVARIATE_NAMES: tuple[str, ...] = ("sex", "gestational_age", "num_DWI_artifact")
N_COVARIATES = len(COVARIATE_NAMES)
SEX_MAP: dict[str, float] = {"Female": 0.0, "Male": 1.0}

# Vineland V06 standard-score columns used as multi-output regression targets.
VINELAND_COLUMNS_V06: tuple[str, ...] = (
    "V06 Vineland,adapt_behave_comp_STD_SCORE",
    "V06 Vineland,communication_STD_SCORE",
    "V06 Vineland,daily_living_skills_STD_SCORE",
    "V06 Vineland,motor_skills_STD_SCORE",
    "V06 Vineland,socialization_STD_SCORE",
)
VINELAND_SHORT_NAMES: tuple[str, ...] = (
    "ABC", "Communication", "DailyLiving", "Motor", "Socialization",
)

# Default file conventions used by the level VTK output of
# build_neighborhood_graph.py.
DEFAULT_LEVEL_BASE = "FiberAxisProfiles_merged_imputed_neighborhood"
# We default to the coarser pair (L2, L3) because:
#   * L0 (4766 nodes) and L1 (2434 nodes) inflate the per-fold compute and the
#     per-node interpretation sweep without a measurable AUC gain in practice.
#   * L2 (1267) + L3 (682) still cover ~2k nodes -- plenty of spatial resolution
#     for per-tract attribution -- while running ~5x faster.
# Users wanting maximum spatial detail can re-enable L0/L1 via --levels.
DEFAULT_LEVELS = (2, 3)


def level_vtk_path(base_dir: Path, level: int, base: str = DEFAULT_LEVEL_BASE) -> Path:
    return base_dir / f"{base}_L{level}.vtk"


def level_npz_path(base_dir: Path, level: int, base: str = DEFAULT_LEVEL_BASE) -> Path:
    return base_dir / f"{base}_L{level}.npz"


def level_indices_path(base_dir: Path, level: int, base: str = DEFAULT_LEVEL_BASE) -> Path:
    return base_dir / f"{base}_L{level}.indices.txt"


# Hyperparameter defaults locked in by user experiments.
DEFAULT_SIREN_EPOCHS = 200
DEFAULT_SIREN_OMEGA0 = 10.0

# Cross-validation defaults.
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 5
DEFAULT_OUTER_REPEATS = 3
DEFAULT_SEED = 0
