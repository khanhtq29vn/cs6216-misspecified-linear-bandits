"""
Bandit Algorithms for Realizable and Misspecified Linear Bandits.

Implements four canonical algorithms studied in our CS6216 lecture notes:
1. StandardUCB: Classical unstructured Upper Confidence Bound (Lattimore & Szepesvari, Ch. 7).
2. LinUCB (OFUL): Optimism in the Face of Uncertainty for Linear Bandits
   (Li et al., 2010; Abbasi-Yadkori et al., 2011; Lattimore & Szepesvari, Ch. 19).
3. RobustLinearBanditRLB: Two-phase Linearity Hypothesis-Testing Bandit
   (Ghosh, Chowdhury, & Gopalan, AAAI 2017, Algorithm 1).
4. PhasedEliminationBandit: G-Optimal Design Phased Elimination for Linear Bandits
   (Lattimore & Szepesvari, Ch. 21; Liu, Qiao, Yin, Bogunovic, & Wang, 2025, Algorithm 2).
"""

from typing import Optional
import numpy as np
from src.utils import (
    batch_mahalanobis_norm,
    compute_g_optimal_design,
    select_linearly_independent_basis,
    sherman_morrison_update,
)


class StandardUCB:
    """
    Unstructured k-armed Upper Confidence Bound (UCB) learner.

    Ignores arm features X and treats all k arms independently, achieving
    O(sqrt(k n log n)) minimax regret and O((k log n) / Delta) instance-dependent
    regret regardless of linear model misspecification.
    """

    def __init__(self, k: int, sigma: float = 0.25, c_bonus: float = 2.0) -> None:
        self.k = k
        self.sigma = sigma
        self.c_bonus = c_bonus
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.k, dtype=int)
        self.emp_means = np.zeros(self.k, dtype=float)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        # Pull each arm at least once
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        log_term = np.log(max(2, self.t))
        bonuses = np.sqrt(
            (self.c_bonus * (self.sigma**2) * log_term) / self.counts
        )
        ucb_indices = self.emp_means + bonuses
        return int(np.argmax(ucb_indices))

    def update(self, arm_idx: int, reward: float) -> None:
        self.counts[arm_idx] += 1
        n_i = self.counts[arm_idx]
        self.emp_means[arm_idx] += (reward - self.emp_means[arm_idx]) / float(n_i)


class LinUCB:
    """
    LinUCB / OFUL (Optimism in the Face of Uncertainty for Linear Bandits).

    At round t, maintains regularized design matrix V_{t-1} = lambda I_d + sum_{s=1}^{t-1} X_s X_s^T
    and ridge regression estimator theta_hat_{t-1} = V_{t-1}^{-1} sum_{s=1}^{t-1} Y_s X_s,
    selecting:
        X_t = argmax_{x in X} { <x, theta_hat_{t-1}> + sqrt(beta_t) ||x||_{V_{t-1}^{-1}} }.
    """

    def __init__(
        self,
        features: np.ndarray,
        sigma: float = 0.25,
        reg_lambda: float = 1.0,
        delta: float = 0.05,
        theta_norm_bound: float = 1.0,
        alpha_scale: float = 0.5,
    ) -> None:
        self.features = features.astype(float)
        self.k, self.d = self.features.shape
        self.sigma = sigma
        self.reg_lambda = reg_lambda
        self.delta = delta
        self.S = theta_norm_bound
        self.L = float(np.max(np.linalg.norm(self.features, axis=1)))
        self.alpha_scale = alpha_scale
        self.reset()

    def reset(self) -> None:
        self.V_inv = (1.0 / self.reg_lambda) * np.eye(self.d, dtype=float)
        self.b = np.zeros(self.d, dtype=float)
        self.theta_hat = np.zeros(self.d, dtype=float)
        self.t = 0

    def _confidence_radius(self, t: int) -> float:
        """
        Self-normalized confidence ellipsoid radius sqrt(beta_t) from
        Abbasi-Yadkori et al. (2011, Theorem 2) and Lattimore & Szepesvari (Theorem 20.5).
        """
        log_det_bound = self.d * np.log(
            1.0 + (t * (self.L**2)) / (self.reg_lambda * self.d)
        )
        sqrt_beta = np.sqrt(self.reg_lambda) * self.S + self.sigma * np.sqrt(
            2.0 * np.log(1.0 / self.delta) + log_det_bound
        )
        return self.alpha_scale * float(sqrt_beta)

    def select_arm(self) -> int:
        self.t += 1
        radius = self._confidence_radius(self.t)
        pred_means = self.features @ self.theta_hat
        widths = batch_mahalanobis_norm(self.features, self.V_inv)
        ucb_scores = pred_means + radius * widths
        return int(np.argmax(ucb_scores))

    def update(self, arm_idx: int, reward: float) -> None:
        x = self.features[arm_idx]
        self.V_inv = sherman_morrison_update(self.V_inv, x)
        self.b += reward * x
        self.theta_hat = self.V_inv @ self.b


class RobustLinearBanditRLB:
    """
    Robust Linear Bandit (RLB) via Linearity Hypothesis Testing
    (Ghosh, Chowdhury, & Gopalan, AAAI 2017, Section 4 / Algorithm 1).

    Mechanism:
    - Phase 1 (Pure Exploration & Subspace Linearity Test):
      Selects a basis B of d linearly independent arms plus diagnostic test arms
      outside B, pulling each tau times. Let mu_hat_B be the empirical means on B.
      Under the null hypothesis H_0 (exact linearity eps = 0), the unique linear
      parameter is theta_B = X_B^{-1} mu_B, so any non-basis arm j must have mean
      <x_j, X_B^{-1} mu_B>.
      RLB compares the extrapolated linear estimate <x_j, X_B^{-1} mu_hat_B> against
      the direct empirical mean mu_hat_j. If the discrepancy exceeds the high-probability
      confidence threshold, RLB rejects H_0 (detecting model misspecification H_1).
    - Phase 2 (Adaptive Exploitation):
      * If H_0 is accepted: executes LinUCB (warm-started with Phase 1 data) to achieve
        O(d sqrt(n)) regret.
      * If H_0 is rejected: switches to unstructured StandardUCB (warm-started with
        Phase 1 data) to guarantee sublinear O(sqrt(k n)) regret and avoid the Omega(n)
        linear failure mode of LinUCB.
    """

    def __init__(
        self,
        features: np.ndarray,
        sigma: float = 0.25,
        tau_per_arm: int = 20,
        delta: float = 0.05,
        reg_lambda: float = 1.0,
        alpha_scale: float = 0.5,
        num_test_arms: int = 3,
    ) -> None:
        self.features = features.astype(float)
        self.k, self.d = self.features.shape
        self.sigma = sigma
        self.tau = tau_per_arm
        self.delta = delta
        self.reg_lambda = reg_lambda
        self.alpha_scale = alpha_scale

        basis_idx, non_basis_idx = select_linearly_independent_basis(
            self.features, self.d
        )
        self.basis_indices = basis_idx
        # Include non-basis arms for hypothesis testing (in Ghosh et al., 1 arm suffices
        # when deviations are non-sparse; testing a small diagnostic set covers sparse cases)
        if len(non_basis_idx) > 0:
            # Prioritize arms with largest projection onto the basis or last index
            selected_non_basis = non_basis_idx[-min(num_test_arms, len(non_basis_idx)) :]
            self.test_indices = np.array(selected_non_basis, dtype=int)
        else:
            self.test_indices = np.array([], dtype=int)

        self.phase1_arms = np.concatenate([self.basis_indices, self.test_indices])
        self.total_phase1_rounds = len(self.phase1_arms) * self.tau
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.phase1_complete = False
        self.rejected_H0 = False
        self.max_test_stat = 0.0
        self.test_threshold = 0.0

        self.linucb = LinUCB(
            features=self.features,
            sigma=self.sigma,
            reg_lambda=self.reg_lambda,
            delta=self.delta,
            alpha_scale=self.alpha_scale,
        )
        self.ucb = StandardUCB(k=self.k, sigma=self.sigma)

    def _run_hypothesis_test(self) -> None:
        """
        Evaluate whether the confidence intervals of linear basis extrapolation I_e
        and direct empirical sampling I_s intersect on the test arms.
        """
        if len(self.test_indices) == 0 or len(self.basis_indices) < self.d:
            self.rejected_H0 = False
            return

        X_B = self.features[self.basis_indices]  # shape (d, d)
        X_B_inv = np.linalg.pinv(X_B)
        mu_hat_B = self.ucb.emp_means[self.basis_indices]
        theta_B_hat = X_B_inv @ mu_hat_B

        max_ratio = 0.0
        for j in self.test_indices:
            x_j = self.features[j]
            # Coefficient vector c_j = (X_B^T)^{-1} x_j such that x_j = X_B^T c_j
            c_j = X_B_inv.T @ x_j
            pred_linear = float(np.dot(x_j, theta_B_hat))
            emp_direct = float(self.ucb.emp_means[j])
            diff = abs(pred_linear - emp_direct)

            # Variance of ( <c_j, mu_hat_B> - mu_hat_j ) is sigma^2 * (||c_j||_2^2 + 1) / tau
            std_scale = self.sigma * np.sqrt(
                (float(np.dot(c_j, c_j)) + 1.0) / float(self.tau)
            )
            threshold_j = std_scale * np.sqrt(
                2.0 * np.log(4.0 * len(self.test_indices) / self.delta)
            )
            if threshold_j > 0:
                ratio = diff / threshold_j
                if ratio > max_ratio:
                    max_ratio = ratio
                    self.max_test_stat = diff
                    self.test_threshold = threshold_j

            if diff > threshold_j:
                self.rejected_H0 = True

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.total_phase1_rounds:
            idx_in_list = (self.t - 1) // self.tau
            return int(self.phase1_arms[idx_in_list])

        if not self.phase1_complete:
            self._run_hypothesis_test()
            self.phase1_complete = True

        if self.rejected_H0:
            return self.ucb.select_arm()
        return self.linucb.select_arm()

    def update(self, arm_idx: int, reward: float) -> None:
        # Warm-start both learners during Phase 1 so whichever is chosen in Phase 2
        # utilizes all collected observations without throwing away data
        self.ucb.update(arm_idx, reward)
        self.linucb.update(arm_idx, reward)


class PhasedEliminationBandit:
    """
    G-Optimal Design Phased Elimination for Linear Bandits
    (Lattimore & Szepesvari, Ch. 21; Liu, Qiao, Yin, Bogunovic, & Wang, 2025, Algorithm 2).

    Why Phased Elimination Tolerates Larger rho-GAM (rho = O(1/sqrt(d))) than LinUCB:
    In adaptive algorithms like LinUCB, the design matrix V_t depends on past rewards
    and can become skewed toward arms with positive misspecification bias. In contrast,
    Phased Elimination computes a deterministic Kiefer-Wolfowitz G-optimal design pi_l
    over the active arm set A_l at the start of each phase l, ensuring uniform directional
    control:
        max_{x in A_l} ||x||_{V_l^{-1}} <= sqrt(d / N_l).
    Consequently, the misspecification bias in the phase-l least-squares estimate is
    uniformly bounded by O(sqrt(d) * rho * max_{x in A_l} Delta(x)), allowing accurate
    elimination of high-gap arms in early phases and progressively tighter estimation
    in later phases!
    """

    def __init__(
        self,
        features: np.ndarray,
        sigma: float = 0.25,
        delta: float = 0.05,
        phase_scale: float = 0.35,
    ) -> None:
        self.features = features.astype(float)
        self.k, self.d = self.features.shape
        self.sigma = sigma
        self.delta = delta
        self.phase_scale = phase_scale
        self.reset()

    def reset(self) -> None:
        self.active_arms = np.arange(self.k, dtype=int)
        self.phase_idx = 0
        self.phase_queue: list[int] = []
        self.phase_arm_pulls: list[int] = []
        self.phase_rewards: list[float] = []
        self._start_new_phase()

    def _start_new_phase(self) -> None:
        self.phase_idx += 1
        self.phase_arm_pulls = []
        self.phase_rewards = []

        if len(self.active_arms) == 1:
            # Only 1 arm remains active: pull it indefinitely
            self.phase_queue = [int(self.active_arms[0])] * 10000
            return

        eps_l = 2.0 ** (-self.phase_idx)
        active_X = self.features[self.active_arms]
        pi_l, _ = compute_g_optimal_design(active_X)

        # Total pulls in phase l scales as d * sigma^2 * log(k * l^2 / delta) / eps_l^2
        log_factor = np.log(
            (self.k * self.phase_idx * (self.phase_idx + 1)) / self.delta
        )
        base_budget = (
            self.phase_scale
            * 2.0
            * self.d
            * (self.sigma**2)
            * log_factor
            / (eps_l**2)
        )

        queue = []
        for local_idx, global_arm in enumerate(self.active_arms):
            if pi_l[local_idx] > 1e-5:
                m_i = max(1, int(np.ceil(pi_l[local_idx] * base_budget)))
                queue.extend([int(global_arm)] * m_i)

        self.phase_queue = queue

    def _finish_phase_and_eliminate(self) -> None:
        if len(self.active_arms) <= 1:
            return

        pulled_indices = np.array(self.phase_arm_pulls, dtype=int)
        rewards = np.array(self.phase_rewards, dtype=float)
        X_pulled = self.features[pulled_indices]

        # Compute fresh phase-l least-squares estimator using Moore-Penrose pseudoinverse
        V_l = X_pulled.T @ X_pulled + 1e-6 * np.eye(self.d)
        b_l = X_pulled.T @ rewards
        theta_hat_l = np.linalg.pinv(V_l, rcond=1e-8) @ b_l

        active_X = self.features[self.active_arms]
        est_means = active_X @ theta_hat_l
        max_est_mean = float(np.max(est_means))

        eps_l = 2.0 ** (-self.phase_idx)
        # Elimination threshold 2 * eps_l
        keep_mask = (max_est_mean - est_means) <= (2.0 * eps_l)
        if np.any(keep_mask):
            self.active_arms = self.active_arms[keep_mask]

    def select_arm(self) -> int:
        if len(self.phase_queue) == 0:
            self._finish_phase_and_eliminate()
            self._start_new_phase()
        return self.phase_queue.pop(0)

    def update(self, arm_idx: int, reward: float) -> None:
        self.phase_arm_pulls.append(arm_idx)
        self.phase_rewards.append(reward)