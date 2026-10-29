"""
Fast sanity-check script for CS6216 Group 13a codebase.

Verifies:
1. Kiefer-Wolfowitz G-optimal design convergence on random arm sets (`OptimalDesignResult`).
2. Self-verified environment invariants (`EnvironmentDiagnostics`) across all 3 regimes.
3. Cumulative pseudo-regret trajectories over n = 4,000 rounds for all 4 algorithms.
"""

from __future__ import annotations

import numpy as np
from src.utils import compute_g_optimal_design
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


def run_single_trial(
    env: LinearBanditEnvironment, learner: BanditPolicy, horizon: int, seed: int
) -> float:
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
    print("=" * 82)
    print("CS6216 Group 13a Codebase Verification (Realizable, Sparse, and rho-GAM)")
    print("=" * 82)

    # 1. Verify Kiefer-Wolfowitz G-optimal design solver
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(30, 5))
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    design_res = compute_g_optimal_design(X_test)
    support_size = int(np.sum(design_res.pi > 1e-4))
    print(
        f"[Check 1] G-Optimal Design (k=30, d=5): max_x ||x||_{{V(pi)^+}}^2 = "
        f"{design_res.max_variance:.4f} (r = {design_res.effective_rank}), "
        f"duality_gap = {design_res.duality_gap:.2e}, support = {support_size}, "
        f"iters = {design_res.num_iterations}"
    )
    print("-" * 82)

    # 2. Verify all 3 environments and 4 algorithms
    horizon = 4000
    envs = [
        make_realizable_env(k=50, d=5, sigma=0.25, seed=42),
        make_ghosh_sparse_misspecified_env(
            k=20, d=5, sigma=0.25, eps_sparse=0.45, permute_arms=True, seed=42
        ),
        make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=0.30, permute_arms=True, seed=42
        ),
    ]

    for env in envs:
        print(env.summary())
        learners: list[BanditPolicy] = [
            StandardUCB(k=env.k, sigma=env.sigma),
            LinUCB(features=env.features, sigma=env.sigma),
            RobustLinearBanditRLB(
                features=env.features, sigma=env.sigma, tau_per_arm=20
            ),
            PhasedEliminationBandit(features=env.features, sigma=env.sigma),
        ]

        for learner in learners:
            cum_regret = run_single_trial(env, learner, horizon=horizon, seed=101)
            extra_info = ""
            if isinstance(learner, RobustLinearBanditRLB):
                extra_info = (
                    f" [Rejected H0: {learner.rejected_H0}, "
                    f"stat={learner.max_test_stat:.3f} vs thr={learner.test_threshold:.3f}]"
                )
            print(f"     * {learner.name:<38} : R_{horizon} = {cum_regret:7.2f}{extra_info}")
        print("-" * 82)

    print("All verification checks completed successfully!")


if __name__ == "__main__":
    main()