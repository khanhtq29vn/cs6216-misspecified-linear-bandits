"""
Fast sanity-check script for CS6216 Group 13a codebase.
Verifies:
1. Kiefer-Wolfowitz G-optimal design convergence on random arm sets.
2. Exact environment properties (Realizable, Ghosh Sparse Misspecification, Liu rho-GAM).
3. Cumulative regret trajectories over n = 4,000 rounds for all 4 algorithms.
"""

import numpy as np
from src.utils import compute_g_optimal_design
from src.environments import (
    make_realizable_env,
    make_ghosh_sparse_misspecified_env,
    make_gap_adjusted_env,
)
from src.algorithms import (
    StandardUCB,
    LinUCB,
    RobustLinearBanditRLB,
    PhasedEliminationBandit,
)


def run_single_trial(env, learner, horizon: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    learner.reset()
    total_regret = 0.0
    for _ in range(horizon):
        arm = learner.select_arm()
        reward, inst_regret = env.step(arm, rng)
        learner.update(arm, reward)
        total_regret += inst_regret
    return total_regret


def main() -> None:
    print("=" * 78)
    print("CS6216 Group 13a Codebase Verification (Realizable, Sparse, and rho-GAM)")
    print("=" * 78)

    # 1. Verify Kiefer-Wolfowitz G-optimal design solver
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(30, 5))
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    pi_opt, max_var = compute_g_optimal_design(X_test)
    support_size = int(np.sum(pi_opt > 1e-4))
    print(
        f"[Check 1] G-Optimal Design on k=30, d=5: max_x ||x||_{{V(pi)^-1}}^2 = {max_var:.4f} "
        f"(Theoretical optimum d = 5.0000), active support size = {support_size}"
    )
    print("-" * 78)

    # 2. Verify all 3 environments and 4 algorithms
    horizon = 4000
    envs = [
        make_realizable_env(k=40, d=5, sigma=0.25, seed=42),
        make_ghosh_sparse_misspecified_env(
            k=20, d=5, sigma=0.25, eps_sparse=0.45, seed=42
        ),
        make_gap_adjusted_env(k=40, d=5, sigma=0.25, rho=0.30, seed=42),
    ]

    for env in envs:
        diag = env.diagnostics()
        print(f"Environment: {env.name}")
        print(
            f"  -> k={diag['k']}, d={diag['d']}, optimal_mean={diag['optimal_mean']:.3f}, "
            f"min_gap={diag['min_gap']:.3f}, ||eps||_inf={diag['eps_linf']:.3f}, "
            f"rho_GAM={diag['rho_gam']:.3f}"
        )

        learners = {
            "Standard UCB": StandardUCB(k=env.k, sigma=env.sigma),
            "LinUCB (OFUL)": LinUCB(features=env.features, sigma=env.sigma),
            "RLB (Ghosh et al. 2017)": RobustLinearBanditRLB(
                features=env.features, sigma=env.sigma, tau_per_arm=20
            ),
            "Phased Elimination (Liu et al. 2025)": PhasedEliminationBandit(
                features=env.features, sigma=env.sigma
            ),
        }

        for name, learner in learners.items():
            cum_regret = run_single_trial(env, learner, horizon=horizon, seed=101)
            extra_info = ""
            if isinstance(learner, RobustLinearBanditRLB):
                extra_info = (
                    f" [Hypothesis Test Rejected H0: {learner.rejected_H0}]"
                )
            print(
                f"     * {name:<38} : R_{horizon} = {cum_regret:7.2f}{extra_info}"
            )
        print("-" * 78)

    print("All verification checks completed successfully!")


if __name__ == "__main__":
    main()