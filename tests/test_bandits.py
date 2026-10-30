"""
Comprehensive mathematical, algorithmic, and audit-regression unit test suite
for CS6216 Group 13a.

Run from the project root via:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
import numpy as np

from src.utils import (
    batch_mahalanobis_norm,
    compute_g_optimal_design,
    select_well_conditioned_basis,
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

        with self.assertRaises(ValueError):
            sherman_morrison_update(np.array([[1.0, 2.0], [0.0, 1.0]]), np.ones(2))

    def test_batch_mahalanobis_norm(self) -> None:
        """Verify vectorized Mahalanobis norm computation and PSD validation."""
        rng = np.random.default_rng(456)
        k, d = 25, 5
        X = rng.normal(size=(k, d))
        M = rng.normal(size=(d, d))
        A = M @ M.T + 0.5 * np.eye(d)

        norms_vec = batch_mahalanobis_norm(X, A)
        norms_loop = np.array([np.sqrt(float(x.T @ A @ x)) for x in X])
        np.testing.assert_allclose(norms_vec, norms_loop, rtol=1e-13, atol=1e-13)

        with self.assertRaises(ValueError):
            batch_mahalanobis_norm(X, -np.eye(d))

    def test_kiefer_wolfowitz_g_optimal_design_full_and_low_rank(self) -> None:
        """
        Verify Theorem 2.5 (Kiefer-Wolfowitz Equivalence) and Caratheodory support bound:
        - Independently verify max_{x in A} ||x||_{V(pi)^+}^2 <= rank(span(A)) * (1 + tol)
        - Verify |supp(pi)| <= r(r + 1) / 2.
        """
        rng = np.random.default_rng(789)
        X_full = rng.normal(size=(30, 5))
        X_full /= np.linalg.norm(X_full, axis=1, keepdims=True)
        res_full = compute_g_optimal_design(X_full, tol=1e-4)

        self.assertTrue(res_full.converged)
        self.assertEqual(res_full.effective_rank, 5)
        self.assertAlmostEqual(float(np.sum(res_full.pi)), 1.0, places=12)
        self.assertTrue(np.all(res_full.pi >= 0.0))
        self.assertLessEqual(res_full.max_variance, 5.0 * 1.001)
        self.assertLessEqual(int(np.sum(res_full.pi > 1e-10)), (5 * 6) // 2)

        # Independent unregularized Gram matrix check
        V_gram = (X_full.T * res_full.pi) @ X_full
        V_pinv = np.linalg.pinv(V_gram, rcond=1e-12, hermitian=True)
        indep_max_var = float(np.max(batch_mahalanobis_norm(X_full, V_pinv) ** 2))
        self.assertAlmostEqual(res_full.max_variance, indep_max_var, places=7)

        X_sub = X_full[:3]
        res_sub = compute_g_optimal_design(X_sub, tol=1e-4)
        self.assertTrue(res_sub.converged)
        self.assertEqual(res_sub.effective_rank, 3)
        self.assertLessEqual(res_sub.max_variance, 3.0 * 1.001)

    def test_g_optimal_design_scale_invariance_and_zero_arm_audit_f4(self) -> None:
        """
        Audit F4 Regression: Verify that `1e-7 * X` produces the exact same G-optimal
        design and leverage `max_variance <= 2.0001` as unscaled `X`, and handle zero arms.
        """
        X_base = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.8]], dtype=np.float64)
        X_tiny = 1e-7 * X_base

        res_base = compute_g_optimal_design(X_base, tol=1e-5)
        res_tiny = compute_g_optimal_design(X_tiny, tol=1e-5)

        self.assertTrue(res_base.converged)
        self.assertTrue(res_tiny.converged)
        self.assertEqual(res_tiny.effective_rank, 2)
        self.assertLessEqual(res_tiny.max_variance, 2.0 * (1.0 + 1e-5))
        np.testing.assert_allclose(res_base.pi, res_tiny.pi, atol=1e-10)

        # Single zero arm should have effective_rank = 0 and max_variance = 0.0
        res_zero = compute_g_optimal_design(np.zeros((1, 4), dtype=np.float64))
        self.assertEqual(res_zero.effective_rank, 0)
        self.assertEqual(res_zero.max_variance, 0.0)


class TestBanditEnvironments(unittest.TestCase):
    """Verify mathematical invariants of all three synthetic bandit regimes."""

    def test_environment_immutability_and_audit_f8_optimal_arm_check(self) -> None:
        """
        Audit F8 Regression: Verify read-only views, rejection of nonzero error on any
        optimal arm when `expected_rho_gam` is set, and `rho_gam_ratio == inf` when
        optimal arm carries misspecification.
        """
        env = make_realizable_env(k=20, d=5, seed=1)
        with self.assertRaises(ValueError):
            env.features[0, 0] = 999.0

        # Audit F8 counterexample: linear means (0, 0.5, 1), true means (0, 1, 1),
        # errors (0, 0.5, 0) with claimed rho=0 must raise AssertionError!
        X_f8 = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float64)
        theta_f8 = np.array([1.0, 0.0], dtype=np.float64)
        eps_f8 = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            LinearBanditEnvironment(
                name="F8 Counterexample",
                features=X_f8,
                theta_star=theta_f8,
                misspecification=eps_f8,
                sigma=0.1,
                expected_rho_gam=0.0,
            )

        # Ghosh sparse environment with eps_k = 0.45 violates optimality preservation -> rho_GAM = inf
        env_ghosh = make_ghosh_sparse_misspecified_env(k=20, d=5, eps_sparse=0.45, seed=42)
        self.assertTrue(np.isinf(env_ghosh.diagnostics().rho_gam_ratio))

        # Verify d=1 support in make_realizable_env
        env_d1 = make_realizable_env(k=5, d=1, min_gap=0.2, seed=7)
        self.assertEqual(env_d1.d, 1)

    def test_ghosh_phase_transition_threshold(self) -> None:
        """
        Verify Equation (12) of `main.tex`: phase transition at eps_k^* = 0.215.
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
        Verify Definition 4.1 and Proposition 4.2 on `make_gap_adjusted_env`.
        """
        rho = 0.30
        env = make_gap_adjusted_env(k=40, d=5, rho=rho, permute_arms=True, seed=42)
        diag = env.diagnostics()

        self.assertTrue(diag.preserves_linear_optimum)
        self.assertAlmostEqual(diag.rho_gam_ratio, rho, places=10)
        self.assertAlmostEqual(diag.eps_linf_norm, 0.35, places=10)


class TestBanditPolicies(unittest.TestCase):
    """Verify theoretical behavior and audit edge cases of all four bandit policies."""

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

    def test_audit_f2_rlb_child_clock_sync_and_f6_high_dim_pe(self) -> None:
        """
        Audit F2 & F6 Regressions:
        1. Verify that after Phase 1 (`t == 160`), RLB immediately completes its test
           and synchronizes `self.ucb.t == 160` and `self.linucb.t == 160`.
        2. Verify that `PhasedEliminationBandit(np.eye(101))` always returns a valid
           arm index in `0..100` instead of out-of-bounds `101`.
        """
        env = make_ghosh_sparse_misspecified_env(k=20, d=5, eps_sparse=0.45, seed=42)
        rlb = RobustLinearBanditRLB(features=env.features, sigma=env.sigma, tau_per_arm=20)
        rng = np.random.default_rng(99)
        for _ in range(rlb.total_phase1_rounds):
            arm = rlb.select_arm()
            rew, _ = env.step(arm, rng)
            rlb.update(arm, rew)

        self.assertEqual(rlb.t, 160)
        self.assertEqual(rlb.ucb.t, 160)
        self.assertEqual(rlb.linucb.t, 160)
        self.assertTrue(rlb.phase1_complete)
        self.assertTrue(rlb.rejected_H0)

        # High-dimensional identity matrix k = 101, d = 101
        pe_101 = PhasedEliminationBandit(features=np.eye(101, dtype=np.float64))
        chosen_arm = pe_101.select_arm()
        self.assertTrue(0 <= chosen_arm < 101)

    def test_permutation_invariance_of_rlb_basis_and_test_arms(self) -> None:
        """
        Audit F9 Verification: Verify that `RobustLinearBanditRLB` selects the exact
        same physical basis vectors and the exact same physical test vectors under
        arbitrary row permutations.
        """
        env_unpermuted = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.45, permute_arms=False, seed=42
        )
        env_permuted = make_ghosh_sparse_misspecified_env(
            k=20, d=5, eps_sparse=0.45, permute_arms=True, seed=42
        )

        rlb_unperm = RobustLinearBanditRLB(features=env_unpermuted.features, sigma=0.25)
        rlb_perm = RobustLinearBanditRLB(features=env_permuted.features, sigma=0.25)

        basis_vecs_unperm = env_unpermuted.features[rlb_unperm.basis_indices]
        basis_vecs_perm = env_permuted.features[rlb_perm.basis_indices]
        test_vecs_unperm = env_unpermuted.features[rlb_unperm.test_indices]
        test_vecs_perm = env_permuted.features[rlb_perm.test_indices]

        np.testing.assert_allclose(basis_vecs_unperm, basis_vecs_perm, atol=1e-12)
        np.testing.assert_allclose(test_vecs_unperm, test_vecs_perm, atol=1e-12)
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
        self.assertLess(r_lin, 0.45 * r_ucb)
        self.assertLess(r_pe, 0.55 * r_ucb)
        self.assertLess(r_rlb, 0.65 * r_ucb)

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
        self.assertLess(r_rlb, 160.0)

    def test_regime3_liu_rho_gam_no_regret_recovery(self) -> None:
        """Verify LinUCB and Phased Elimination under rho-GAM (rho = 0.30)."""
        env = make_gap_adjusted_env(
            k=40, d=5, sigma=0.25, rho=0.30, permute_arms=True, seed=42
        )
        n = 3000
        ucb = StandardUCB(k=env.k, sigma=env.sigma)
        linucb = LinUCB(features=env.features, sigma=env.sigma)
        rlb = RobustLinearBanditRLB(features=env.features, sigma=env.sigma, tau_per_arm=20)
        pe = PhasedEliminationBandit(features=env.features, sigma=env.sigma)

        r_ucb = self._simulate(env, ucb, n, seed=303)
        r_lin = self._simulate(env, linucb, n, seed=303)
        r_rlb = self._simulate(env, rlb, n, seed=303)
        r_pe = self._simulate(env, pe, n, seed=303)

        self.assertTrue(rlb.rejected_H0)
        self.assertLess(r_lin, 45.0)
        self.assertLess(r_pe, 1.15 * r_ucb)
        self.assertLess(r_lin, 0.45 * r_rlb)


if __name__ == "__main__":
    unittest.main()