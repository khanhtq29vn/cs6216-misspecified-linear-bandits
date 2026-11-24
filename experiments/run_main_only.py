"""Run the three main experiments without the auxiliary parameter sweeps.

This mode is useful for validating the lecture-notes summary table before
committing time to every sweep; it writes an explicitly incomplete archive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from experiments.provenance import provenance_fields, atomic_savez

from src.environments import (
    make_gap_adjusted_env,
    make_ghosh_sparse_misspecified_env,
    make_realizable_env,
)
from experiments.run_experiments import (
    ALGORITHM_NAMES, export_latex_summary_table, simulate_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main-table experiments only")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/main_only"))
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing main-only results")
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon < 1:
        parser.error("--seeds and --horizon must be positive")
    results: dict[str, np.ndarray] = {
        "meta_complete": np.array([0], dtype=np.int8),
        "meta_main_seeds": np.array([args.seeds], dtype=np.int64),
        "meta_main_horizon": np.array([args.horizon], dtype=np.int64),
        "meta_sweep_seeds": np.array([0], dtype=np.int64),
        "meta_sweep_horizon": np.array([0], dtype=np.int64),
        "meta_method_names": np.asarray(ALGORITHM_NAMES, dtype=str),
        "meta_python_version": np.asarray([sys.version.split()[0]]),
        "meta_numpy_version": np.asarray([np.__version__]),
    }
    results.update(provenance_fields())
    file_path = args.output_dir / "results_main_only.npz"
    table_path = args.output_dir / "table_main_only.tex"
    if (file_path.exists() or table_path.exists()) and not args.overwrite:
        parser.error("Existing output detected; choose a new --output-dir or --overwrite")
    configurations = (
        ("exp1", lambda seed: make_realizable_env(k=50, d=5, sigma=0.25, seed=seed), 100),
        ("exp2", lambda seed: make_ghosh_sparse_misspecified_env(
            k=20, d=5, sigma=0.25, eps_sparse=0.45, seed=seed), 2000),
        ("exp3", lambda seed: make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=0.30, seed=seed), 4000),
    )
    for label, factory, base_seed in configurations:
        print(f"Evaluating {label}: {args.seeds} seeds, horizon {args.horizon}", flush=True)
        grid, means, ses, rejection, raw, rejection_flags = simulate_trajectories(
            factory, horizon=args.horizon, n_seeds=args.seeds,
            base_seed=base_seed, return_raw=True,
        )
        results["t_grid"] = grid
        key = "exp1_rlb_reject_rate" if label == "exp1" else f"{label}_rlb_reject_rate_main"
        results[key] = np.array([rejection])
        results[f"{label}_rlb_reject_trials"] = rejection_flags
        for name in ALGORITHM_NAMES:
            results[f"{label}_traj_{name}_mean"] = means[name]
            results[f"{label}_traj_{name}_trials"] = raw[name]
            results[f"{label}_traj_{name}_se"] = ses[name]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(file_path, results)
    export_latex_summary_table(results, str(table_path))
    print("Results (main table only; no auxiliary sweeps):")
    for name in ALGORITHM_NAMES:
        out = []
        for label in ("exp1", "exp2", "exp3"):
            m = float(results[f"{label}_traj_{name}_mean"][-1])
            se = float(results[f"{label}_traj_{name}_se"][-1])
            out.append(f"{m:.1f} ± {se:.1f}")
        print(f"  {name}: " + " | ".join(out))
    print(f"Saved {file_path}")


if __name__ == "__main__":
    main()
