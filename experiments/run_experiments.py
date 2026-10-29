"""
Multi-seed Monte Carlo simulation runner for CS6216 Group 13a.

Executes four structured experimental benchmarks:
1. Experiment 1 (Realizable Regime, eps = 0):
   - 1a: Cumulative regret trajectories over horizon n = 6,000 (k = 50, d = 5).
   - 1b: Final regret scaling across number of arms k in {10, 25, 50, 100, 150, 200}.
2. Experiment 2 (Sparse Subspace Misspecification, Ghosh et al., AAAI 2017):
   - 2a: Cumulative regret trajectories over n = 6,000 for eps_k = 0.45 (k = 20, d = 5).
   - 2b: Phase transition across sparse deviation magnitude eps_k in [0.0, 0.50] and
         empirical hypothesis-test power P(Reject H_0) of RLB.
3. Experiment 3 (rho-Gap-Adjusted Misspecification, Liu, Bogunovic et al., 2025):
   - 3a: Cumulative regret trajectories over n = 6,000 for rho = 0.30 (k = 40, d = 5).
   - 3b: Final regret across misspecification ratio rho in [0.05, 0.60] under skewed
         suboptimal bias, comparing LinUCB vs. G-Optimal Phased Elimination.
4. Experiment 4 (Structural & Geometric Diagnostics):
   - 4a: Pointwise true rewards f_0(x_i), linear approximation <x_i, w^*>, and the
         rho-GAM cone vs. uniform l_inf tube.
   - 4b: Trajectory of the Upper Confidence Bound index UCB_k(t) of the optimal arm
         under unmodified LinUCB vs. Basis Arm 1 (permutation-invariant).

Outputs:
- `experiments/results_summary.npz`: Compressed NumPy archive of all aggregated arrays.
- `experiments/table_results.tex`: Formatted LaTeX summary table for Section 5 of the notes.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Dict, List, Tuple
import numpy as np

from src.environments import (
    LinearBanditEnvironment,
    make_gap_adjusted_env,
    make_ghosh_sparse_misspecified_env,
    make_realizable_env,
)
from src.algorithms import (
    BanditPolicy,
    LinUCB,
    PhasedEliminationBandit,
    RobustLinearBanditRLB,
    StandardUCB,
)


ALGORITHM_NAMES: List[str] = [
    "Standard UCB",
    "LinUCB (OFUL)",
    "RLB (Ghosh et al. 2017)",
    "Phased Elimination (Liu et al. 2025)",
]


def build_learners(env: LinearBanditEnvironment) -> Dict[str, BanditPolicy]:
    """
    Instantiate the four benchmark algorithms with consistent, verified hyperparameters.
    """
    return {
        "Standard UCB": StandardUCB(k=env.k, sigma=env.sigma, c_bonus=2.0),
        "LinUCB (OFUL)": LinUCB(
            features=env.features,
            sigma=env.sigma,
            reg_lambda=1.0,
            delta=0.05,
            confidence_scale=0.50,
        ),
        "RLB (Ghosh et al. 2017)": RobustLinearBanditRLB(
            features=env.features,
            sigma=env.sigma,
            tau_per_arm=20,
            delta=0.05,
            reg_lambda=1.0,
            confidence_scale=0.50,
            num_test_arms=3,
        ),
        "Phased Elimination (Liu et al. 2025)": PhasedEliminationBandit(
            features=env.features,
            sigma=env.sigma,
            delta=0.05,
            phase_scale=0.12,
            eps_init=0.25,
        ),
    }


def simulate_trajectories(
    env_factory: Callable[[int], LinearBanditEnvironment],
    horizon: int,
    n_seeds: int,
    subsample_step: int = 20,
    base_seed: int = 1000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
    """
    Run Monte Carlo simulations across `n_seeds` independent trials and record
    subsampled cumulative pseudo-regret trajectories.

    Returns
    -------
    t_grid : np.ndarray
        Time steps at which cumulative regret is recorded.
    mean_trajs : Dict[str, np.ndarray]
        Mean cumulative regret curve for each algorithm.
    se_trajs : Dict[str, np.ndarray]
        Standard error (std / sqrt(n_seeds)) curve for each algorithm.
    rlb_reject_rate : float
        Fraction of trials in which RLB rejected the linearity null hypothesis H_0.
    """
    t_grid = np.arange(subsample_step, horizon + 1, subsample_step, dtype=int)
    n_points = len(t_grid)

    all_trajs = {
        name: np.zeros((n_seeds, n_points), dtype=np.float64)
        for name in ALGORITHM_NAMES
    }
    rlb_rejections = 0

    for s_idx in range(n_seeds):
        env_seed = base_seed + 97 * s_idx
        env = env_factory(env_seed)
        learners = build_learners(env)

        for alg_idx, name in enumerate(ALGORITHM_NAMES):
            learner = learners[name]
            rng = np.random.default_rng(base_seed + 1009 * s_idx + 31 * alg_idx)
            learner.reset()

            cum_regret = 0.0
            rec_ptr = 0
            for t in range(1, horizon + 1):
                arm = learner.select_arm()
                reward, inst_regret = env.step(arm, rng)
                learner.update(arm, reward)
                cum_regret += inst_regret

                if t == t_grid[rec_ptr]:
                    all_trajs[name][s_idx, rec_ptr] = cum_regret
                    rec_ptr += 1

            if isinstance(learner, RobustLinearBanditRLB) and learner.rejected_H0:
                rlb_rejections += 1

    mean_trajs = {name: np.mean(all_trajs[name], axis=0) for name in ALGORITHM_NAMES}
    se_trajs = {
        name: np.std(all_trajs[name], axis=0, ddof=1) / np.sqrt(max(1, n_seeds))
        if n_seeds > 1
        else np.zeros(n_points, dtype=np.float64)
        for name in ALGORITHM_NAMES
    }
    rlb_reject_rate = float(rlb_rejections) / float(n_seeds)
    return t_grid, mean_trajs, se_trajs, rlb_reject_rate


def compute_structural_diagnostics() -> Dict[str, np.ndarray]:
    """
    Generate exact geometric and algorithmic diagnostic data for Figure 4:
    - Panel 4a: Pointwise reward profile f_0(x_i) vs. linear prediction <x_i, w^*>
                and the rho-GAM cone in Environment 3 (rho = 0.30).
    - Panel 4b: Trajectory of LinUCB's upper confidence bound index UCB_k(t) on
                the sparse optimal arm vs. suboptimal Basis Arm 1 in Environment 2
                (completely invariant to arm index permutation).
    """
    # Panel 4a data: rho-GAM cone structure
    env_gam = make_gap_adjusted_env(
        k=40, d=5, sigma=0.25, rho=0.30, skew_suboptimal_bias=False, seed=42
    )
    order = np.argsort(env_gam.linear_means)
    lin_sorted = env_gam.linear_means[order]
    true_sorted = env_gam.true_means[order]
    f_star = env_gam.optimal_mean
    rho_val = 0.30

    lin_gap_grid = np.linspace(0.0, 1.55, 200)
    lin_val_grid = f_star - lin_gap_grid
    gam_upper_cone = f_star - lin_gap_grid / (1.0 + rho_val)
    gam_lower_cone = f_star - lin_gap_grid / (1.0 - rho_val)
    eps_inf = float(np.max(np.abs(env_gam.misspecification)))

    # Panel 4b data: LinUCB UCB index collapse in Ghosh et al. (2017) environment
    env_ghosh = make_ghosh_sparse_misspecified_env(
        k=20, d=5, sigma=0.25, eps_sparse=0.45, permute_arms=True, seed=42
    )
    horizon_diag = 1500
    linucb = LinUCB(
        features=env_ghosh.features,
        sigma=env_ghosh.sigma,
        reg_lambda=1.0,
        delta=0.05,
        confidence_scale=0.50,
    )
    rng = np.random.default_rng(2026)

    ucb_arm0_trace = np.zeros(horizon_diag, dtype=np.float64)
    ucb_opt_trace = np.zeros(horizon_diag, dtype=np.float64)
    mean_opt_pred_trace = np.zeros(horizon_diag, dtype=np.float64)

    opt_idx = (
        int(env_ghosh.corrupted_arm_idx)
        if env_ghosh.corrupted_arm_idx is not None
        else env_ghosh.optimal_arm
    )
    arm1_idx = (
        int(env_ghosh.basis_arm_1_idx)
        if env_ghosh.basis_arm_1_idx is not None
        else 0
    )

    for t in range(1, horizon_diag + 1):
        ucb_scores, preds, _ = linucb.compute_ucb_scores(t)

        ucb_arm0_trace[t - 1] = ucb_scores[arm1_idx]
        ucb_opt_trace[t - 1] = ucb_scores[opt_idx]
        mean_opt_pred_trace[t - 1] = preds[opt_idx]

        chosen_arm = int(np.argmax(ucb_scores))
        reward, _ = env_ghosh.step(chosen_arm, rng)
        linucb.update(chosen_arm, reward)

    return {
        "diag_lin_sorted": lin_sorted,
        "diag_true_sorted": true_sorted,
        "diag_lin_val_grid": lin_val_grid,
        "diag_gam_upper_cone": gam_upper_cone,
        "diag_gam_lower_cone": gam_lower_cone,
        "diag_eps_inf": np.array([eps_inf]),
        "diag_t_trace": np.arange(1, horizon_diag + 1, dtype=int),
        "diag_ucb_arm0": ucb_arm0_trace,
        "diag_ucb_opt": ucb_opt_trace,
        "diag_pred_opt": mean_opt_pred_trace,
        "diag_true_opt_mean": np.array([env_ghosh.true_means[opt_idx]]),
        "diag_true_arm0_mean": np.array([env_ghosh.true_means[arm1_idx]]),
        "diag_extrap_opt_mean": np.array([env_ghosh.linear_means[opt_idx]]),
    }


def export_latex_summary_table(
    results: Dict[str, np.ndarray], out_tex_path: str
) -> None:
    """
    Generate a clean LaTeX booktabs table summarizing final cumulative regret R_n
    (mean +/- standard error) and RLB rejection rates across the three regimes.
    """
    rej1 = float(results["exp1_rlb_reject_rate"][0]) * 100.0
    rej2 = float(results["exp2_rlb_reject_rate_main"][0]) * 100.0
    rej3 = float(results["exp3_rlb_reject_rate_main"][0]) * 100.0

    lines = [
        "% Auto-generated by experiments/run_experiments.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "\\textbf{Algorithm} & \\textbf{Regime 1: Realizable} & \\textbf{Regime 2: Sparse Misspec.} & \\textbf{Regime 3: $\\rho$-GAM} \\\\",
        " & ($\\varepsilon = 0,\\; k=50,\\; d=5$) & ($\\varepsilon_k = 0.45,\\; k=20,\\; d=5$) & ($\\rho = 0.30,\\; k=40,\\; d=5$) \\\\ \\midrule",
    ]

    for name in ALGORITHM_NAMES:
        r1_m = float(results[f"exp1_traj_{name}_mean"][-1])
        r1_s = float(results[f"exp1_traj_{name}_se"][-1])
        r2_m = float(results[f"exp2_traj_{name}_mean"][-1])
        r2_s = float(results[f"exp2_traj_{name}_se"][-1])
        r3_m = float(results[f"exp3_traj_{name}_mean"][-1])
        r3_s = float(results[f"exp3_traj_{name}_se"][-1])

        if "RLB" in name:
            row = (
                f"{name} & ${r1_m:.1f} \\pm {r1_s:.1f}$ {{\\scriptsize({rej1:.0f}\\% rej.\\ $H_0$)}} & "
                f"${r2_m:.1f} \\pm {r2_s:.1f}$ {{\\scriptsize({rej2:.0f}\\% rej.\\ $H_0$)}} & "
                f"${r3_m:.1f} \\pm {r3_s:.1f}$ {{\\scriptsize({rej3:.0f}\\% rej.\\ $H_0$)}} \\\\"
            )
        else:
            row = (
                f"{name} & ${r1_m:.1f} \\pm {r1_s:.1f}$ & "
                f"${r2_m:.1f} \\pm {r2_s:.1f}$ & "
                f"${r3_m:.1f} \\pm {r3_s:.1f}$ \\\\"
            )
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Empirical cumulative pseudo-regret $R_n$ at horizon $n = 6{,}000$ (mean $\\pm$ standard error over Monte Carlo trials) across the three misspecification regimes.}",
            "\\label{tab:empirical_regret}",
            "\\end{table}",
        ]
    )

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-seed Monte Carlo experiments for CS6216 Group 13a."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast 5-seed evaluation (default: 20 seeds).",
    )
    args = parser.parse_args()

    n_seeds_main = 5 if args.quick else 20
    n_seeds_sweep = 5 if args.quick else 15
    horizon_main = 6000
    horizon_sweep = 4000

    start_time = time.time()
    print("=" * 84)
    print(
        f"Running CS6216 Group 13a Monte Carlo Experiments "
        f"(n_seeds={n_seeds_main}, horizon={horizon_main})"
    )
    print("=" * 84)

    results: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # Experiment 1a: Realizable Regime Regret Trajectories (k = 50, d = 5)
    # -------------------------------------------------------------------------
    print("[1/6] Running Experiment 1a: Realizable Linear Bandit trajectories...")
    t_grid, m1, se1, rlb_rej1 = simulate_trajectories(
        env_factory=lambda s: make_realizable_env(k=50, d=5, sigma=0.25, seed=s),
        horizon=horizon_main,
        n_seeds=n_seeds_main,
        base_seed=100,
    )
    results["t_grid"] = t_grid
    results["exp1_rlb_reject_rate"] = np.array([rlb_rej1])
    for name in ALGORITHM_NAMES:
        results[f"exp1_traj_{name}_mean"] = m1[name]
        results[f"exp1_traj_{name}_se"] = se1[name]

    # -------------------------------------------------------------------------
    # Experiment 1b: Scaling with Number of Arms k in Realizable Regime
    # -------------------------------------------------------------------------
    print("[2/6] Running Experiment 1b: Scaling across arm count k in {10..200}...")
    k_grid = np.array([10, 25, 50, 100, 150, 200], dtype=int)
    results["exp1_k_grid"] = k_grid
    for name in ALGORITHM_NAMES:
        results[f"exp1_ksweep_{name}_mean"] = np.zeros(len(k_grid), dtype=np.float64)
        results[f"exp1_ksweep_{name}_se"] = np.zeros(len(k_grid), dtype=np.float64)

    for idx_k, k_val in enumerate(k_grid):
        _, mk, sek, _ = simulate_trajectories(
            env_factory=lambda s, kv=int(k_val): make_realizable_env(
                k=kv, d=5, sigma=0.25, seed=s
            ),
            horizon=horizon_sweep,
            n_seeds=n_seeds_sweep,
            subsample_step=horizon_sweep,
            base_seed=500 + 50 * idx_k,
        )
        for name in ALGORITHM_NAMES:
            results[f"exp1_ksweep_{name}_mean"][idx_k] = mk[name][-1]
            results[f"exp1_ksweep_{name}_se"][idx_k] = sek[name][-1]

    # -------------------------------------------------------------------------
    # Experiment 2a: Ghosh et al. (2017) Sparse Misspecification Trajectories
    # -------------------------------------------------------------------------
    print("[3/6] Running Experiment 2a: Sparse Misspecified Bandit trajectories...")
    _, m2, se2, rlb_rej2 = simulate_trajectories(
        env_factory=lambda s: make_ghosh_sparse_misspecified_env(
            k=20, d=5, sigma=0.25, eps_sparse=0.45, permute_arms=True, seed=s
        ),
        horizon=horizon_main,
        n_seeds=n_seeds_main,
        base_seed=2000,
    )
    results["exp2_rlb_reject_rate_main"] = np.array([rlb_rej2])
    for name in ALGORITHM_NAMES:
        results[f"exp2_traj_{name}_mean"] = m2[name]
        results[f"exp2_traj_{name}_se"] = se2[name]

    # -------------------------------------------------------------------------
    # Experiment 2b: Sweep across Sparse Deviation Magnitude eps_k
    # -------------------------------------------------------------------------
    print("[4/6] Running Experiment 2b: Phase transition across eps_k in [0.0, 0.50]...")
    eps_grid = np.array([0.00, 0.10, 0.20, 0.28, 0.35, 0.42, 0.50], dtype=np.float64)
    results["exp2_eps_grid"] = eps_grid
    results["exp2_rlb_reject_rate"] = np.zeros(len(eps_grid), dtype=np.float64)
    for name in ALGORITHM_NAMES:
        results[f"exp2_sweep_{name}_mean"] = np.zeros(len(eps_grid), dtype=np.float64)
        results[f"exp2_sweep_{name}_se"] = np.zeros(len(eps_grid), dtype=np.float64)

    for idx_e, eps_val in enumerate(eps_grid):
        _, me, see, rej_e = simulate_trajectories(
            env_factory=lambda s, ev=float(eps_val): make_ghosh_sparse_misspecified_env(
                k=20, d=5, sigma=0.25, eps_sparse=ev, permute_arms=True, seed=s
            ),
            horizon=horizon_sweep,
            n_seeds=n_seeds_sweep,
            subsample_step=horizon_sweep,
            base_seed=3000 + 70 * idx_e,
        )
        results["exp2_rlb_reject_rate"][idx_e] = rej_e
        for name in ALGORITHM_NAMES:
            results[f"exp2_sweep_{name}_mean"][idx_e] = me[name][-1]
            results[f"exp2_sweep_{name}_se"][idx_e] = see[name][-1]

    # -------------------------------------------------------------------------
    # Experiment 3a: Liu et al. (2025) rho-GAM Regret Trajectories (rho = 0.30)
    # -------------------------------------------------------------------------
    print("[5/6] Running Experiment 3a: rho-GAM Bandit trajectories (rho = 0.30)...")
    _, m3, se3, rlb_rej3 = simulate_trajectories(
        env_factory=lambda s: make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=0.30, skew_suboptimal_bias=False, permute_arms=True, seed=s
        ),
        horizon=horizon_main,
        n_seeds=n_seeds_main,
        base_seed=4000,
    )
    results["exp3_rlb_reject_rate_main"] = np.array([rlb_rej3])
    for name in ALGORITHM_NAMES:
        results[f"exp3_traj_{name}_mean"] = m3[name]
        results[f"exp3_traj_{name}_se"] = se3[name]

    # -------------------------------------------------------------------------
    # Experiment 3b: Sweep across rho in [0.05, 0.60] under Skewed Suboptimal Bias
    # -------------------------------------------------------------------------
    print("[6/6] Running Experiment 3b: Sweep across rho in [0.05, 0.60]...")
    rho_grid = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.52, 0.60], dtype=np.float64)
    results["exp3_rho_grid"] = rho_grid
    results["exp3_eps_linf_grid"] = np.zeros(len(rho_grid), dtype=np.float64)
    for name in ALGORITHM_NAMES:
        results[f"exp3_sweep_{name}_mean"] = np.zeros(len(rho_grid), dtype=np.float64)
        results[f"exp3_sweep_{name}_se"] = np.zeros(len(rho_grid), dtype=np.float64)

    for idx_r, rho_val in enumerate(rho_grid):
        sample_env = make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=float(rho_val), skew_suboptimal_bias=True, permute_arms=True, seed=42
        )
        results["exp3_eps_linf_grid"][idx_r] = sample_env.diagnostics()["eps_linf"]

        _, mr, ser, _ = simulate_trajectories(
            env_factory=lambda s, rv=float(rho_val): make_gap_adjusted_env(
                k=40, d=5, sigma=0.25, rho=rv, skew_suboptimal_bias=True, permute_arms=True, seed=s
            ),
            horizon=horizon_sweep,
            n_seeds=n_seeds_sweep,
            subsample_step=horizon_sweep,
            base_seed=5000 + 90 * idx_r,
        )
        for name in ALGORITHM_NAMES:
            results[f"exp3_sweep_{name}_mean"][idx_r] = mr[name][-1]
            results[f"exp3_sweep_{name}_se"][idx_r] = ser[name][-1]

    # Add structural diagnostic arrays for Figure 4
    diag_dict = compute_structural_diagnostics()
    results.update(diag_dict)

    # Save compressed archive and LaTeX table
    os.makedirs("experiments", exist_ok=True)
    npz_path = os.path.join("experiments", "results_summary.npz")
    tex_path = os.path.join("experiments", "table_results.tex")
    np.savez_compressed(npz_path, **results)
    export_latex_summary_table(results, tex_path)

    elapsed = time.time() - start_time
    print("-" * 84)
    print(f"Summary of Final Cumulative Regret R_n at n = {horizon_main}:")
    print(
        f"{'Algorithm':<38} | {'Realizable':>14} | {'Sparse Misspec':>14} | {'rho-GAM (0.30)':>14}"
    )
    print("-" * 84)
    for name in ALGORITHM_NAMES:
        r1 = f"{results[f'exp1_traj_{name}_mean'][-1]:6.1f} ± {results[f'exp1_traj_{name}_se'][-1]:4.1f}"
        r2 = f"{results[f'exp2_traj_{name}_mean'][-1]:6.1f} ± {results[f'exp2_traj_{name}_se'][-1]:4.1f}"
        r3 = f"{results[f'exp3_traj_{name}_mean'][-1]:6.1f} ± {results[f'exp3_traj_{name}_se'][-1]:4.1f}"
        print(f"{name:<38} | {r1:>14} | {r2:>14} | {r3:>14}")
    print("-" * 84)
    print(
        f"RLB H0 Rejection Rates: Regime 1 = {rlb_rej1*100:.0f}%, "
        f"Regime 2 = {rlb_rej2*100:.0f}%, Regime 3 = {rlb_rej3*100:.0f}%"
    )
    print(f"Saved numerical arrays to: {npz_path}")
    print(f"Saved LaTeX table to     : {tex_path}")
    print(f"Completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()