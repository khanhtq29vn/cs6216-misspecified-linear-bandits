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

__all__ = [
    "FloatArray",
    "IntArray",
    "OptimalDesignResult",
    "sherman_morrison_update",
    "batch_mahalanobis_norm",
    "select_well_conditioned_basis",
    "compute_g_optimal_design",
]
