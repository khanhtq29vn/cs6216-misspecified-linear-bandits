"""
Experimental evaluation and visualization suite for CS6216 Group 13a:
Contextual and Linear Bandits under Model Misspecification.
"""

from experiments.run_experiments import (
    ALGORITHM_NAMES,
    build_learners,
    compute_structural_diagnostics,
    export_latex_summary_table,
    simulate_trajectories,
)

__all__ = [
    "ALGORITHM_NAMES",
    "build_learners",
    "simulate_trajectories",
    "compute_structural_diagnostics",
    "export_latex_summary_table",
]