"""
Core simulation library for CS6216 Group 13a:
Contextual and Linear Bandits under Model Misspecification.
"""

from src.utils import (
    FloatArray,
    IntArray,
    OptimalDesignResult,
    batch_mahalanobis_norm,
    compute_g_optimal_design,
    select_well_conditioned_basis,
    sherman_morrison_update,
)
from src.environments import (
    EnvironmentDiagnostics,
    LinearBanditEnvironment,
    make_gap_adjusted_env,
    make_ghosh_sparse_misspecified_env,
    make_realizable_env,
)

__all__ = [
    "FloatArray",
    "IntArray",
    "OptimalDesignResult",
    "sherman_morrison_update",
    "batch_mahalanobis_norm",
    "select_well_conditioned_basis",
    "compute_g_optimal_design",
    "EnvironmentDiagnostics",
    "LinearBanditEnvironment",
    "make_realizable_env",
    "make_ghosh_sparse_misspecified_env",
    "make_gap_adjusted_env",
]