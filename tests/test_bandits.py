"""
Comprehensive mathematical and algorithmic unit test suite for CS6216 Group 13a.

Run from the project root via:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
import numpy as np

from src.utils import (
    batch_mahalanobis_norm,
    compute_g_optimal_design,
    sherman_morrison_update,
)
from src.environments import (
    LinearBanditEnvironment,
    make_gap_adjusted_env,
    make_ghosh_sparse_misspecified_env,
    make_realizable_env,
)
from src.algorithms import (
    LinUCB,
    PhasedEliminationBandit,
    RobustLinearBanditRLB,
    StandardUCB,
)


class TestNumericalUtilities(unittest.TestCase):
    """Verify numerical linear algebra routines and Kiefer-Wolfowitz optimal design."""

    def test_sherman_morrison_exactness_and_symmetry(self) -> None:
        """
        Verify that 500 sequential rank-1 Sherman-Morrison updates match explicit
        matrix inversion `np.linalg.inv(V_t)` within 1e-11 and preserve exact symmetry.
        """
        rng = np.random.default_rng(123)
        d = 5
        reg_lambda = 1.25
        V = reg_lambda * np.eye(d, dtype=np.float64)
        V_inv = (1.0 / reg_lambda) * np.eye(d, dtype=np.float64)

        for _ in range(500):
            x = rng.normal(size=d)
            x /= np.linalg.norm(x)
            V += np.outer(x, x)
            V_inv = sherman_morrison_update(V_inv, x)

        explicit_inv = np.linalg.inv(V)
        frob_err = float(np.linalg.norm(V_inv - explicit_inv, ord="fro"))
        sym_err = float(np.linalg.norm(V_inv - V_inv.T, ord="fro"))

        self.assertLess(frob_err, 1e-11)
        self.assertEqual(sym_err, 0.0)

    def test_batch_mahalanobis_norm(self) -> None:
        """Verify vectorized Mahalanobis norm computation against loop evaluation."""
        rng = np.random.default_rng(456)
        k, d = 25, 5
        X = rng.normal(size=(k, d))
        M = rng.normal(size=(d, d))
        A = M @ M.T + 0.5 * np.eye(d)

        norms_vec = batch_mahalanobis_norm(X, A)
        norms_loop = np.array([np.sqrt(float(x.T @ A @ x)) for x in X])
        np.testing.assert_allclose(norms_vec, norms_loop, rtol=1e-13, atol=1e-13)

    def test_kiefer_wolfowitz_g_optimal_design_full_and_low_rank(self) -> None:
        """
        Verify Theorem 2.5 (Kiefer-Wolfowitz Equivalence):
        max_{x in A} ||x||_{V(pi)^+}^2 <= rank(span(A)) * (1 + tol)
        both for full-rank sets (k=30, d=5, r=5) and rank-deficient subsets (k=3, d=5, r=3).
        """
        rng = np.random.default_rng(789)
        X_full = rng.normal(size=(30, 5))
        X_full /= np.linalg.norm(X_full, axis=1, keepdims=True)
        res_full = compute_g_optimal_design(X_full, tol=1e-4)

        self.assertEqual(res_full.effective_rank, 5)
        self.assertAlmostEqual(float(np.sum(res_full.pi)), 1.0, places=12)
        self.assertTrue(np.all(res_full.pi >= 0.0))
        self.assertLessEqual(res_full.max_variance, 5.0 * 1.001)

        X_sub = X_full[:3]
        res_sub = compute_g_optimal_design(X_sub, tol=1e-4)
        self.assertEqual(res_sub.effective_rank, 3)
        self.assertAlmostEqual(float(np.sum(res_sub.pi)), 1.0, places=12)
        self.assertLessEqual(res_sub.max_variance, 3.0 * 1.001)


class TestBanditEnvironments(unittest.TestCase):
    """Verify mathematical invariants of all three synthetic bandit regimes."""

    def test_environment_immutability_and_invariant_checks(self) -> None:
        """Verify that environment feature arrays are read-only and enforce bounds."""
        env = make_realizable_env(k=20, d=5, seed=1)
        with self.assertRaises(ValueError):
            env.features[0, 0] = 999.0

        bad_X = np.ones((5, 3), dtype=np.float64)
        with self.assertRaises(AssertionError):
            LinearBanditEnvironment(
                name="Invalid Norm Env",
                features=bad_X,
                theta_star=np.array([1.0, 0.0, 0.0]),
                misspecification=np.zeros(5),
                sigma=0.1,
                feature_norm_bound=1.0,
            )

    def test_ghosh_phase_transition_threshold(self) -> None:
        """
        Verify Equation (12) of `main.tex`:
        In Ghosh et al.'s sparse environment, Basis Arm 1 has mu_1 = 0.70 and the
        corrupted arm has mu_k^{lin} = 0.485, so the phase transition occurs at
        eps_k^* = 0.70 - 0.485 = 0.215.
        """
        env_below = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.15, permute_arms=True, seed=42
        )
        self.assertEqual(env_below.optimal_arm, env_below.basis_arm_1_idx)
        self.assertAlmostEqual(env_below.optimal_mean, 0.70, places=10)

        env_above = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.45, permute_arms=True, seed=42
        )
        self.assertEqual(env_above.optimal_arm, env_above.corrupted_arm_idx)
        self.assertAlmostEqual(env_above.optimal_mean, 0.935, places=10)

    def test_liu_rho_gam_gap_sandwiching_and_optimality(self) -> None:
        """
        Verify Definition 4.1 and Proposition 4.2 on `make_gap_adjusted_env`:
        - Exact optimality preservation: argmax f_{w^*} == argmax f_0
        - Exact relative ratio: |eps_i| / Delta_i == rho for all suboptimal arms
        - Multiplicative gap sandwiching: (1 - rho) Delta_i <= Delta_i^{lin} <= (1 + rho) Delta_i
        """
        rho = 0.30
        env = make_gap_adjusted_env(k=40, d=5, rho=rho, permute_arms=True, seed=42)
        diag = env.diagnostics()

        self.assertTrue(diag.preserves_linear_optimum)
        self.assertAlmostEqual(diag.rho_gam_ratio, rho, places=10)
        self.assertAlmostEqual(diag.eps_linf_norm, 0.35, places=10)

        non_opt = env.suboptimality_gaps > 1e-10
        true_gaps = env.suboptimality_gaps[non_opt]
        lin_gaps = env.optimal_mean - env.linear_means[non_opt]
        np.testing.assert_allclose(
            np.abs(env.misspecification[non_opt]) / true_gaps,
            rho,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(np.all(lin_gaps >= (1.0 - rho) * true_gaps - 1e-12))
        self.assertTrue(np.all(lin_gaps <= (1.0 + rho) * true_gaps + 1e-12))


class TestBanditPolicies(unittest.TestCase):
    """Verify theoretical behavior of all four bandit policies across regimes."""

    @staticmethod
    def _simulate(env: LinearBanditEnvironment, policy, horizon: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        policy.reset()
        regret = 0.0
        for _ in range(horizon):
            arm = policy.select_arm()
            reward, inst_reg = env.step(arm, rng)
            policy.update(arm, reward)
            regret += inst_reg
        return regret

    def test_permutation_invariance_of_rlb_basis_and_test_arms(self) -> None:
        """
        Verify that `RobustLinearBanditRLB` selects the exact same basis vectors and
        includes the corrupted arm in `test_indices` regardless of arm permutation.
        """
        env_unpermuted = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.45, permute_arms=False, seed=42
        )
        env_permuted = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.45, permute_arms=True, seed=42
        )

        rlb_unperm = RobustLinearBanditRLB(features=env_unpermuted.features, sigma=0.25)
        rlb_perm = RobustLinearBanditRLB(features=env_permuted.features, sigma=0.25)

        self.assertIn(env_unpermuted.corrupted_arm_idx, rlb_unperm.test_indices)
        self.assertIn(env_permuted.corrupted_arm_idx, rlb_perm.test_indices)

    def test_regime1_realizable_sublinear_regret(self) -> None:
        """Verify Theorem 2.4 & Theorem 3.3(i) in Regime 1 (eps = 0)."""
        env = make_realizable_env(k=50, d=5, sigma=0.25, seed=42)
        n = 3000
        ucb = StandardUCB(k=env.k, sigma=env.sigma)
        linucb = LinUCB(features=env.features, sigma=env.sigma)
        rlb = RobustLinearBanditRLB(features=env.features, sigma=env.sigma, tau_per_arm=20)
        pe = PhasedEliminationBandit(features=env.features, sigma=env.sigma)

        r_ucb = self._simulate(env, ucb, n, seed=101)
        r_lin = self._simulate(env, linucb, n, seed=101)
        r_rlb = self._simulate(env, rlb, n, seed=101)
        r_pe = self._simulate(env, pe, n, seed=101)

        self.assertFalse(rlb.rejected_H0)
        self.assertLess(r_lin, 0.30 * r_ucb)
        self.assertLess(r_pe, 0.50 * r_ucb)
        self.assertLess(r_rlb, 0.60 * r_ucb)

    def test_regime2_ghosh_sparse_breakdown_and_rlb_rescue(self) -> None:
        """Verify Theorem 3.2 (LinUCB Omega(n) failure) & Theorem 3.3(ii) (RLB rescue)."""
        env = make_ghosh_sparse_misspecified_env(
            k=20, d=5, sigma=0.25, eps_sparse=0.45, permute_arms=True, seed=42
        )
        n = 3000
        linucb = LinUCB(features=env.features, sigma=env.sigma)
        rlb = RobustLinearBanditRLB(features=env.features, sigma=env.sigma, tau_per_arm=20)

        r_lin = self._simulate(env, linucb, n, seed=202)
        r_rlb = self._simulate(env, rlb, n, seed=202)

        self.assertGreater(r_lin, 550.0)
        self.assertTrue(rlb.rejected_H0)
        self.assertLess(r_rlb, 150.0)

    def test_regime3_liu_rho_gam_no_regret_recovery(self) -> None:
        """Verify Theorems 4.3 & 4.4 (LinUCB and Phased Elimination under rho-GAM)."""
        env = make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=0.30, permute_arms=True, seed=42
        )
        n = 3000
        linucb = LinUCB(features=env.features, sigma=env.sigma)
        rlb = RobustLinearBanditRLB(features=env.features, sigma=env.sigma, tau_per_arm=20)
        pe = PhasedEliminationBandit(features=env.features, sigma=env.sigma)

        r_lin = self._simulate(env, linucb, n, seed=303)
        r_rlb = self._simulate(env, rlb, n, seed=303)
        r_pe = self._simulate(env, pe, n, seed=303)

        self.assertTrue(rlb.rejected_H0)
        self.assertLess(r_lin, 45.0)
        self.assertLess(r_pe, 65.0)
        self.assertLess(r_lin, 0.45 * r_rlb)


if __name__ == "__main__":
    unittest.main()