"""Numerical invariants and reproducibility tests independent of regret claims."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.algorithms import PhasedEliminationBandit, RobustLinearBanditRLB
from src.environments import LinearBanditEnvironment, make_realizable_env, make_gap_adjusted_env, make_ghosh_sparse_misspecified_env
from experiments.provenance import SEED_PROTOCOL, _SOURCE_FILES, source_fingerprint
from src.utils import compute_g_optimal_design, select_well_conditioned_basis
from experiments.run_experiments import (
    ALGORITHM_NAMES,
    build_learners,
    export_latex_summary_table,
    simulate_trajectories,
)


class TestPhaseLocalEstimation(unittest.TestCase):
    def test_phase_allocation_is_interleaved(self) -> None:
        features = np.eye(3)
        policy = PhasedEliminationBandit(
            features, sigma=0.25, phase_scale=0.12,
            eps_init=0.25, elimination_multiplier=2.0,
        )
        counts = policy.phase_remaining_pulls.copy()
        self.assertGreaterEqual(int(np.sum(counts)), 6)
        first_three = []
        for _ in range(3):
            arm = policy.select_arm()
            first_three.append(arm)
            policy.update(arm, 0.0)
        self.assertEqual(len(set(first_three)), 3)
        self.assertEqual(int(policy.phase_idx), 1)

    def test_arm_relabeling_does_not_change_physical_sample_sequence(self) -> None:
        # Identical physical arms and identical per-arm noise streams. Pure
        # relabeling must not change the finite-horizon sampling schedule.
        reference = make_realizable_env(k=12, d=3, sigma=0.25,
                                        permute_arms=False, seed=197)
        noise = np.random.default_rng(839).normal(0.0, 0.25, size=(12, 1500))
        physical_histories = []
        for perm_seed in (0, 1, 2):
            permutation = np.random.default_rng(perm_seed).permutation(12)
            env = LinearBanditEnvironment(
                name="permuted", features=reference.features[permutation],
                theta_star=reference.theta_star,
                misspecification=reference.misspecification[permutation],
                sigma=0.25,
            )
            policy = PhasedEliminationBandit(env.features, sigma=0.25,
                                            phase_scale=0.12, eps_init=0.25,
                                            elimination_multiplier=2.0)
            pulls = np.zeros(12, dtype=int)
            physical_actions = []
            for _ in range(750):
                arm = policy.select_arm()
                physical = int(permutation[arm])
                physical_actions.append(physical)
                reward = float(reference.true_means[physical]
                               + noise[physical, pulls[physical]])
                pulls[physical] += 1
                policy.update(arm, reward)
            physical_histories.append(physical_actions)
        self.assertEqual(physical_histories[0], physical_histories[1])
        self.assertEqual(physical_histories[0], physical_histories[2])

    def test_phase_counts_reset_and_global_counts_persist(self) -> None:
        features = np.eye(2, dtype=float)
        learner = PhasedEliminationBandit(
            features, sigma=0, elimination_multiplier=2,
        )
        first_phase_len = int(np.sum(learner.phase_remaining_pulls))
        self.assertGreater(first_phase_len, 0)
        for _ in range(first_phase_len):
            arm = learner.select_arm()
            learner.update(arm, float(arm))
        self.assertEqual(learner.phase_idx, 1)
        np.testing.assert_equal(learner.total_pull_counts.sum(), first_phase_len)
        np.testing.assert_equal(learner.phase_pull_counts.sum(), first_phase_len)
        learner.select_arm()  # finishes phase 1 and begins phase 2 if both active
        # With noiseless zero threshold the smaller-reward arm is eliminated.
        self.assertEqual(len(learner.active_arms), 1)
        np.testing.assert_equal(learner.phase_pull_counts.sum(), 0)
        np.testing.assert_equal(learner.total_pull_counts.sum(), first_phase_len)

    def test_phase_local_ignores_previous_rewards(self) -> None:
        # A large positive offset in phase 1 must not contaminate phase 2 OLS.
        features = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        learner = PhasedEliminationBandit(
            features, sigma=0.25, elimination_multiplier=1000,
        )
        n1 = int(learner.phase_remaining_pulls.sum())
        for _ in range(n1):
            arm = learner.select_arm()
            learner.update(arm, 100.0 if arm == 0 else 0.0)
        self.assertEqual(learner.phase_idx, 1)
        learner.select_arm()  # advances, preserves all arms under huge threshold
        self.assertEqual(learner.phase_idx, 2)
        np.testing.assert_equal(learner.phase_reward_sums, np.zeros(3))
        self.assertGreater(int(learner.total_pull_counts.sum()), int(learner.phase_pull_counts.sum()))
        # Finish pending select before additional actions.
        learner.update(learner._pending_arm, 0.0)
        while learner.phase_idx == 2:
            arm = learner.select_arm()
            learner.update(arm, 0.0)
        self.assertEqual(learner.phase_idx, 3)
        self.assertEqual(learner.active_arms.size, 3)

    def test_update_protocol(self) -> None:
        learner = PhasedEliminationBandit(np.eye(2))
        with self.assertRaises(ValueError):
            learner.update(0, 0.5)
        arm = learner.select_arm()
        with self.assertRaises(RuntimeError):
            learner.select_arm()
        with self.assertRaises(ValueError):
            learner.update(1 - arm, 0.5)
        learner.update(arm, 0.5)

    def test_g_design_certificate_and_batch_rounding(self) -> None:
        learner = PhasedEliminationBandit(
            np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.6]]),
        )
        self.assertTrue(np.any(learner.phase_remaining_pulls > 0))
        self.assertGreaterEqual(int(learner.phase_remaining_pulls.sum()), learner._current_budget)


    def test_tiny_nonzero_features_are_not_treated_as_zero_rank(self) -> None:
        X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        base = compute_g_optimal_design(X, tol=0.05)
        ref_basis, _ = select_well_conditioned_basis(X)
        for factor in (1.0, 1e-12, 1e-16, 1e-100, 1e-300):
            scaled = factor * X
            design = compute_g_optimal_design(scaled, tol=0.05)
            self.assertEqual(design.effective_rank, 2)
            self.assertAlmostEqual(design.max_variance, base.max_variance, places=8)
            np.testing.assert_array_equal(select_well_conditioned_basis(scaled)[0], ref_basis)
            pe = PhasedEliminationBandit(scaled, sigma=0.0)
            self.assertEqual(pe.active_arms.size, 3)
            count = int(pe.phase_remaining_pulls.sum())
            for _ in range(count):
                arm = pe.select_arm()
                pe.update(arm, float(scaled[arm, 0]))
            pe.select_arm()  # must finish the first phase without a false rank failure


class TestDetectionScope(unittest.TestCase):
    def test_unexamined_optimal_arm_can_pass_diagnostic_test(self) -> None:
        X = np.array([[1., 0.], [0., 1.], [.2, .1], [.3, .1], [.4, .1], [.5, .2]])
        policy = RobustLinearBanditRLB(X, sigma=0.0, tau_per_arm=2, num_test_arms=1)
        unseen = next(i for i in range(len(X)) if i not in set(policy.phase1_arms.tolist()))
        linear_means = X @ np.array([0.7, 0.4])
        true_means = linear_means.copy()
        true_means[unseen] = 5.0
        for _ in range(policy.total_phase1_rounds):
            arm = policy.select_arm()
            policy.update(arm, float(true_means[arm]))
        self.assertFalse(policy.rejected_H0)
        self.assertNotIn(unseen, policy.phase1_arms)
        self.assertEqual(np.argmax(true_means), unseen)


class TestExperimentIntegrity(unittest.TestCase):
    def test_plotting_source_participates_in_provenance(self) -> None:
        self.assertIn("experiments/plot_results.py", _SOURCE_FILES)

    def test_rho_sweep_obeys_bounded_gap(self) -> None:
        for rho in (0.0, 0.05, 0.30, 0.60):
            env = make_gap_adjusted_env(k=40, d=5, rho=rho)
            self.assertLessEqual(float(np.max(env.suboptimality_gaps)), 1.0 + 1e-10)
            self.assertLessEqual(env.diagnostics().rho_gam_ratio, rho + 1e-10)

    def test_learner_identifiers_are_consistent(self) -> None:
        env = make_gap_adjusted_env(k=15, d=3, sigma=0.0)
        self.assertEqual(list(build_learners(env)), ALGORITHM_NAMES)

    def test_single_seed_standard_error_is_zero(self) -> None:
        _, means, ses, rejection = simulate_trajectories(
            env_factory=lambda seed: make_gap_adjusted_env(k=8, d=3, sigma=0.0, seed=seed),
            horizon=25, n_seeds=1, subsample_step=10,
        )
        self.assertTrue(0 <= rejection <= 1)
        for name in ALGORITHM_NAMES:
            self.assertEqual(means[name].shape, (3,))
            np.testing.assert_array_equal(ses[name], np.zeros(3))

    def test_generated_table_uses_run_metadata(self) -> None:
        res = {"meta_main_horizon": np.array([73]), "meta_main_seeds": np.array([3])}
        for label in ("exp1", "exp2", "exp3"):
            key = f"{label}_rlb_reject_rate" if label == "exp1" else f"{label}_rlb_reject_rate_main"
            res[key] = np.array([0.25])
            for name in ALGORITHM_NAMES:
                for suffix in ("mean", "se"):
                    res[f"{label}_traj_{name}_{suffix}"] = np.array([1.0])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "summary.tex"
            export_latex_summary_table(res, str(out))
            content = out.read_text()
        self.assertIn("73", content)
        self.assertIn("3 Monte Carlo trials", content)
        self.assertNotIn("20 Monte Carlo", content)

    def test_quick_run_writes_separate_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "quick"
            subprocess.run(
                [sys.executable, "-m", "experiments.run_experiments", "--quick",
                 "--seeds-main", "1", "--seeds-sweep", "1",
                 "--horizon-main", "30", "--horizon-sweep", "20",
                 "--output-dir", str(out)],
                check=True, capture_output=True, text=True,
            )
            with np.load(out / "results_summary.npz", allow_pickle=False) as data:
                self.assertEqual(int(data["meta_main_seeds"][0]), 1)
                self.assertEqual(int(data["meta_main_horizon"][0]), 30)
                self.assertEqual(len(data["meta_method_names"]), 4)
            self.assertIn("1 Monte Carlo trials", (out / "table_results.tex").read_text())


    def test_paired_environment_geometry_and_provenance(self) -> None:
        # A parameter sweep must hold the underlying randomized arm geometry fixed.
        a = make_gap_adjusted_env(rho=0.05, seed=5000)
        b = make_gap_adjusted_env(rho=0.60, seed=5000)
        np.testing.assert_array_equal(a.features, b.features)
        c = make_ghosh_sparse_misspecified_env(eps_sparse=0.0, seed=3000)
        d = make_ghosh_sparse_misspecified_env(eps_sparse=0.50, seed=3000)
        np.testing.assert_array_equal(c.features, d.features)
        self.assertEqual(c.corrupted_arm_idx, d.corrupted_arm_idx)
        self.assertEqual(len(source_fingerprint()), 64)
        self.assertEqual(SEED_PROTOCOL, "paired-parameter-sweeps-v2")

    def test_raw_trials_reproduce_all_summary_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "paired"
            subprocess.run(
                [sys.executable, "-m", "experiments.run_experiments", "--quick",
                 "--seeds-main", "3", "--seeds-sweep", "3",
                 "--horizon-main", "40", "--horizon-sweep", "20",
                 "--output-dir", str(out)],
                check=True, capture_output=True, text=True,
            )
            with np.load(out / "results_summary.npz", allow_pickle=False) as data:
                self.assertEqual(str(data["meta_seed_protocol"][0]), SEED_PROTOCOL)
                self.assertEqual(str(data["meta_source_sha256"][0]), source_fingerprint())
                for regime in ("exp1", "exp2", "exp3"):
                    for method in ALGORITHM_NAMES:
                        raw = data[f"{regime}_traj_{method}_trials"]
                        np.testing.assert_allclose(raw.mean(axis=0), data[f"{regime}_traj_{method}_mean"])
                        np.testing.assert_allclose(raw.std(axis=0, ddof=1) / np.sqrt(3),
                                                   data[f"{regime}_traj_{method}_se"])
                for prefix in ("exp2", "exp3"):
                    for method in ALGORITHM_NAMES:
                        raw = data[f"{prefix}_sweep_{method}_trials"]
                        np.testing.assert_allclose(raw.mean(axis=1), data[f"{prefix}_sweep_{method}_mean"])
                        np.testing.assert_allclose(raw.std(axis=1, ddof=1) / np.sqrt(3),
                                                   data[f"{prefix}_sweep_{method}_se"])
                self.assertEqual(data["exp2_rlb_reject_trials_sweep"].shape, (7,3))
                self.assertEqual(data["exp3_rlb_reject_trials"].shape, (7,3))

            old_bytes = (out / "results_summary.npz").read_bytes()
            proc = subprocess.run(
                [sys.executable, "-m", "experiments.run_experiments", "--quick",
                 "--seeds-main", "3", "--seeds-sweep", "3",
                 "--horizon-main", "40", "--horizon-sweep", "20",
                 "--output-dir", str(out)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Existing output", proc.stderr)
            self.assertEqual((out / "results_summary.npz").read_bytes(), old_bytes)

            # An archive with counterfeit/stale provenance must fail closed.
            with np.load(out / "results_summary.npz", allow_pickle=False) as data:
                stale = {k: data[k] for k in data.files}
            stale["meta_source_sha256"] = np.asarray(["0" * 64])
            np.savez_compressed(out / "results_summary.npz", **stale)
            proc = subprocess.run(
                [sys.executable, "-m", "experiments.run_experiments", "--quick", "--only-exp3b",
                 "--overwrite", "--seeds-main", "3", "--seeds-sweep", "3",
                 "--horizon-main", "40", "--horizon-sweep", "20",
                 "--output-dir", str(out)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("meta_source_sha256", proc.stderr)

    def test_only_rho_sweep_is_repeatable_for_identical_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            args = [sys.executable, "-m", "experiments.run_experiments", "--quick",
                    "--seeds-main", "1", "--seeds-sweep", "1",
                    "--horizon-main", "30", "--horizon-sweep", "20",
                    "--output-dir", str(out)]
            subprocess.run(args, check=True, capture_output=True, text=True)
            with np.load(out / "results_summary.npz", allow_pickle=False) as data:
                before = {k: data[k].copy() for k in data.files}
            subprocess.run(args + ["--only-exp3b", "--overwrite"],
                           check=True, capture_output=True, text=True)
            with np.load(out / "results_summary.npz", allow_pickle=False) as data:
                for key, old in before.items():
                    np.testing.assert_array_equal(old, data[key], err_msg=key)

    def test_main_only_matches_full_runner_for_identical_seeds(self) -> None:
        # The main-table shortcut must use exactly the same configurations and
        # random streams as the first, third, and fifth stages of the full run.
        with tempfile.TemporaryDirectory() as temp:
            full = Path(temp) / "full"
            short = Path(temp) / "main"
            subprocess.run(
                [sys.executable, "-m", "experiments.run_experiments", "--quick",
                 "--seeds-main", "1", "--seeds-sweep", "1",
                 "--horizon-main", "40", "--horizon-sweep", "20",
                 "--output-dir", str(full)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, "-m", "experiments.run_main_only",
                 "--seeds", "1", "--horizon", "40", "--output-dir", str(short)],
                check=True, capture_output=True, text=True,
            )
            with np.load(full / "results_summary.npz", allow_pickle=False) as a, \
                 np.load(short / "results_main_only.npz", allow_pickle=False) as b:
                self.assertEqual(int(a["meta_complete"][0]), 1)
                self.assertEqual(int(b["meta_complete"][0]), 0)
                self.assertIn("meta_git_commit", b.files)
                for regime in ("exp1", "exp2", "exp3"):
                    for method in ALGORITHM_NAMES:
                        for stat in ("mean", "se"):
                            key = f"{regime}_traj_{method}_{stat}"
                            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
