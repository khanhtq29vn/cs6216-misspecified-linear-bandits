"""
Bandit Policies for Realizable and Misspecified Stochastic Linear Bandits.

Implements the four algorithms analyzed in Sections 2, 3, 4, and 5 of our
CS6216 Lecture Notes (`main.tex`):
1. `StandardUCB`: Unstructured k-armed Upper Confidence Bound policy
   (Lattimore & Szepesvari, 2020, Chapter 7).
2. `LinUCB`: Optimism in the Face of Uncertainty for Linear Bandits (OFUL)
   (Li et al., 2010; Abbasi-Yadkori et al., 2011; Theorem 2.1 & Theorem 2.4).
3. `RobustLinearBanditRLB`: Two-phase Subspace Linearity Hypothesis-Testing Bandit
   (Ghosh, Chowdhury, & Gopalan, AAAI 2017; Algorithm 3 & Theorem 3.3).
4. `PhasedEliminationBandit`: Kiefer-Wolfowitz G-Optimal Design Phased Elimination
   (Lattimore & Szepesvari, 2020, Chapter 21; Liu et al., 2025, Theorem 4.4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from scipy.linalg import qr

from src.utils import (
    FloatArray,
    IntArray,
    batch_mahalanobis_norm,
    compute_g_optimal_design,
    select_well_conditioned_basis,
    sherman_morrison_update,
)


class BanditPolicy(ABC):
    """
    Abstract base interface for sequential bandit policies over a finite action set.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the bandit policy."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal sufficient statistics prior to a new Monte Carlo trial."""

    @abstractmethod
    def select_arm(self) -> int:
        """Select an action index in {0, ..., k-1} for the current interaction round."""

    @abstractmethod
    def update(self, arm_idx: int, reward: float) -> None:
        """Update internal statistics given the pulled action `arm_idx` and observed `reward`."""


class StandardUCB(BanditPolicy):
    """
    Unstructured k-armed Upper Confidence Bound (UCB) learner.

    Ignores feature vectors x_i and maintains independent empirical means mu_hat_i(t-1)
    and pull counts T_i(t-1) for each arm i in {0, ..., k-1}:
        UCB_i(t) = mu_hat_i(t-1) + sqrt( (c_bonus * sigma^2 * log t) / T_i(t-1) ).
    Achieves O(sqrt(k n log n)) worst-case regret and O(sum_{Delta_i > 0} (sigma^2 log n) / Delta_i)
    instance-dependent regret regardless of linear model misspecification.
    """

    def __init__(self, k: int, sigma: float = 0.25, c_bonus: float = 2.0) -> None:
        if k <= 0:
            raise ValueError(f"Number of arms k must be positive, got {k}.")
        self._k: int = int(k)
        self._sigma: float = float(sigma)
        self._c_bonus: float = float(c_bonus)
        self.counts: IntArray = np.zeros(self._k, dtype=np.int64)
        self.emp_means: FloatArray = np.zeros(self._k, dtype=np.float64)
        self.t: int = 0
        self.reset()

    @property
    def name(self) -> str:
        return "Standard UCB"

    def reset(self) -> None:
        self.counts = np.zeros(self._k, dtype=np.int64)
        self.emp_means = np.zeros(self._k, dtype=np.float64)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        unpulled = np.where(self.counts == 0)[0]
        if unpulled.size > 0:
            return int(unpulled[0])

        log_term = float(np.log(max(2, self.t)))
        bonuses = np.sqrt(
            (self._c_bonus * (self._sigma**2) * log_term)
            / self.counts.astype(np.float64)
        )
        ucb_indices = self.emp_means + bonuses
        return int(np.argmax(ucb_indices))

    def update(self, arm_idx: int, reward: float) -> None:
        self.counts[arm_idx] += 1
        n_i = int(self.counts[arm_idx])
        self.emp_means[arm_idx] += (float(reward) - self.emp_means[arm_idx]) / float(
            n_i
        )


class LinUCB(BanditPolicy):
    """
    LinUCB / OFUL (Optimism in the Face of Uncertainty for Linear Bandits).

    At the start of round t (after t-1 observations), maintains the regularized inverse
    design matrix V_{t-1}^{-1} = (lambda I_d + sum_{s=1}^{t-1} X_s X_s^T)^{-1} via O(d^2)
    rank-1 Sherman-Morrison updates and the ridge estimator:
        theta_hat_{t-1} = V_{t-1}^{-1} sum_{s=1}^{t-1} Y_s X_s.
    Selects the action maximizing the upper confidence support function (Algorithm 1):
        X_t = argmax_{x in X} { <x, theta_hat_{t-1}> + sqrt(beta_t) ||x||_{V_{t-1}^{-1}} },
    where the theoretical self-normalized confidence radius sqrt(beta_t) from Theorem 2.1
    (Abbasi-Yadkori et al., 2011, Theorem 2) is:
        sqrt(beta_t) = sqrt(lambda) S + sigma * sqrt(2 log(1/delta) + d log(1 + (t-1) L^2 / (lambda d))).
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        reg_lambda: float = 1.0,
        delta: float = 0.05,
        theta_norm_bound: float = 1.0,
        confidence_scale: float = 0.50,
        alpha_scale: Optional[float] = None,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {features.shape}.")
        if reg_lambda <= 0.0:
            raise ValueError(f"reg_lambda must be > 0, got {reg_lambda}.")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must lie in (0, 1), got {delta}.")

        self.features: FloatArray = np.array(features, dtype=np.float64, copy=True)
        self.k, self.d = self.features.shape
        self.sigma: float = float(sigma)
        self.reg_lambda: float = float(reg_lambda)
        self.delta: float = float(delta)
        self.S: float = float(theta_norm_bound)
        self.L: float = float(np.max(np.linalg.norm(self.features, axis=1)))
        self.confidence_scale: float = (
            float(alpha_scale) if alpha_scale is not None else float(confidence_scale)
        )

        self.V_inv: FloatArray = np.eye(self.d, dtype=np.float64)
        self.b: FloatArray = np.zeros(self.d, dtype=np.float64)
        self.theta_hat: FloatArray = np.zeros(self.d, dtype=np.float64)
        self.t: int = 0
        self.reset()

    @property
    def name(self) -> str:
        return "LinUCB (OFUL)"

    def reset(self) -> None:
        self.V_inv = (1.0 / self.reg_lambda) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)
        self.theta_hat = np.zeros(self.d, dtype=np.float64)
        self.t = 0

    def confidence_radius(self, t: int) -> float:
        """
        Compute the self-normalized confidence ellipsoid radius sqrt(beta_t) at the
        start of round t (after n_obs = max(0, t - 1) observations), matching Theorem 2.1.
        """
        n_obs = max(0, int(t) - 1)
        log_det_bound = self.d * np.log(
            1.0 + (n_obs * (self.L**2)) / (self.reg_lambda * self.d)
        )
        sqrt_beta = np.sqrt(self.reg_lambda) * self.S + self.sigma * np.sqrt(
            2.0 * np.log(1.0 / self.delta) + log_det_bound
        )
        return self.confidence_scale * float(sqrt_beta)

    def _confidence_radius(self, t: int) -> float:
        """Backward-compatible alias for `confidence_radius(t)`."""
        return self.confidence_radius(t)

    def compute_ucb_scores(
        self, t: Optional[int] = None
    ) -> Tuple[FloatArray, FloatArray, float]:
        """
        Compute the optimistic UCB index vector, ridge mean predictions, and confidence
        radius at round `t` (defaults to `self.t + 1`).
        """
        round_idx = (self.t + 1) if t is None else int(t)
        radius = self.confidence_radius(round_idx)
        pred_means = self.features @ self.theta_hat
        widths = batch_mahalanobis_norm(self.features, self.V_inv)
        ucb_scores = pred_means + radius * widths
        return ucb_scores, pred_means, radius

    def select_arm(self) -> int:
        self.t += 1
        ucb_scores, _, _ = self.compute_ucb_scores(self.t)
        return int(np.argmax(ucb_scores))

    def update(self, arm_idx: int, reward: float) -> None:
        x = self.features[arm_idx]
        self.V_inv = sherman_morrison_update(self.V_inv, x)
        self.b += float(reward) * x
        self.theta_hat = self.V_inv @ self.b


class RobustLinearBanditRLB(BanditPolicy):
    """
    Robust Linear Bandit (RLB) via Subspace Linearity Hypothesis Testing
    (Ghosh, Chowdhury, & Gopalan, AAAI 2017, Algorithm 1; Section 3.3 & Algorithm 3 in `main.tex`).
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        tau_per_arm: int = 20,
        delta: float = 0.05,
        reg_lambda: float = 1.0,
        confidence_scale: float = 0.50,
        alpha_scale: Optional[float] = None,
        num_test_arms: int = 3,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {features.shape}.")
        self.features: FloatArray = np.array(features, dtype=np.float64, copy=True)
        self.k, self.d = self.features.shape
        self.sigma: float = float(sigma)
        self.tau: int = int(tau_per_arm)
        self.delta: float = float(delta)
        self.reg_lambda: float = float(reg_lambda)
        self.confidence_scale: float = (
            float(alpha_scale) if alpha_scale is not None else float(confidence_scale)
        )

        basis_idx, non_basis_idx = select_well_conditioned_basis(self.features)
        self.basis_indices: IntArray = basis_idx

        self._X_B_cols: FloatArray = self.features[self.basis_indices].T
        self._X_B_pinv: FloatArray = np.linalg.pinv(self._X_B_cols, rcond=1e-10)

        if non_basis_idx.size > 0:
            m = min(int(num_test_arms), int(non_basis_idx.size))
            coords_all = self._X_B_pinv @ self.features[non_basis_idx].T
            coord_norms = np.linalg.norm(coords_all, axis=0)
            order_by_norm = np.argsort(coord_norms, kind="mergesort")
            if m == 1:
                chosen_pos = np.array([non_basis_idx.size - 1], dtype=np.int64)
            else:
                chosen_pos = np.unique(
                    np.linspace(0, non_basis_idx.size - 1, num=m, dtype=np.int64)
                )
            self.test_indices: IntArray = non_basis_idx[order_by_norm[chosen_pos]]
        else:
            self.test_indices = np.array([], dtype=np.int64)

        self.phase1_arms: IntArray = np.concatenate(
            [self.basis_indices, self.test_indices]
        )
        self.total_phase1_rounds: int = int(self.phase1_arms.size) * self.tau

        self.t: int = 0
        self.phase1_complete: bool = False
        self.rejected_H0: bool = False
        self.max_test_stat: float = 0.0
        self.test_threshold: float = 0.0
        self.reset()

    @property
    def name(self) -> str:
        return "RLB (Ghosh et al. 2017)"

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
            confidence_scale=self.confidence_scale,
        )
        self.ucb = StandardUCB(k=self.k, sigma=self.sigma)

    def _run_hypothesis_test(self) -> None:
        """
        Execute the Phase-1 subspace linearity hypothesis test (Equations (7)-(8)
        and Remark 3.4 of `main.tex`).
        """
        if self.test_indices.size == 0 or self.basis_indices.size == 0:
            self.rejected_H0 = False
            return

        mu_hat_B: FloatArray = self.ucb.emp_means[self.basis_indices]
        m = int(self.test_indices.size)
        log_term = float(np.log(2.0 * float(m) / self.delta))

        max_ratio = -1.0
        self.rejected_H0 = False

        for j in self.test_indices:
            x_j = self.features[int(j)]
            c_j = self._X_B_pinv @ x_j
            extrapolated_mean = float(np.dot(c_j, mu_hat_B))
            direct_mean = float(self.ucb.emp_means[int(j)])
            discrepancy = abs(extrapolated_mean - direct_mean)

            variance_proxy = (
                (self.sigma**2)
                * (float(np.dot(c_j, c_j)) + 1.0)
                / float(self.tau)
            )
            threshold_j = float(np.sqrt(2.0 * variance_proxy * log_term))

            if threshold_j > 0.0:
                ratio = discrepancy / threshold_j
                if ratio > max_ratio:
                    max_ratio = ratio
                    self.max_test_stat = discrepancy
                    self.test_threshold = threshold_j

            if discrepancy > threshold_j:
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
        self.ucb.update(arm_idx, reward)
        self.linucb.update(arm_idx, reward)


class PhasedEliminationBandit(BanditPolicy):
    """
    Kiefer-Wolfowitz G-Optimal Design Phased Elimination for Linear Bandits
    (Lattimore & Szepesvari, 2020, Chapter 21; Liu et al., 2025, Algorithm 2 & Theorem 4.4).

    Algorithmic & Statistical Design:
    - Operates in doubling epochs l = 1, 2, ... with tolerance eps_l = eps_0 * 2^{-(l-1)}
      (eps_0 = 0.25, so elimination thresholds 2 * eps_l are 0.50, 0.25, 0.125, 0.0625, ...).
    - At the start of epoch l, solves for the sparse Kiefer-Wolfowitz G-optimal
      distribution pi_l in Delta(A_l) via `compute_g_optimal_design` (Away-Step Frank-Wolfe
      in the r_l-dimensional orthonormal subspace of A_l) so |supp(pi_l)| <= d(d+1)/2.
    - Ensures each support arm x_i in supp(pi_l) reaches the phase-l target sample size
      m_l(x_i) = max(1, round(pi_l(x_i) * N_l)) by allocating incremental pulls
      max(1, m_l(x_i) - T_{active}(x_i)).
    - Active-Subspace Least-Squares Estimation & Arm Elimination:
      Projects surviving arms A_l onto the orthonormal basis Q_l in R^{d x r_l} of span(A_l),
      forms the active-set Gram matrix G_l using ONLY observations from surviving arms in A_l
      (completely purging all observations from previously eliminated high-gap arms),
      computes phi_hat_l = G_l^{-1} b_l, and retains all x in A_l satisfying:
          max_{y in A_l} <y, theta_hat_l> - <x, theta_hat_l> <= 2 * eps_l.
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        delta: float = 0.05,
        phase_scale: float = 0.12,
        eps_init: float = 0.25,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {features.shape}.")
        self.features: FloatArray = np.array(features, dtype=np.float64, copy=True)
        self.k, self.d = self.features.shape
        self.sigma: float = float(sigma)
        self.delta: float = float(delta)
        self.phase_scale: float = float(phase_scale)
        self.eps_init: float = float(eps_init)

        self.active_arms: IntArray = np.arange(self.k, dtype=np.int64)
        self.phase_idx: int = 0
        self.phase_remaining_pulls: IntArray = np.zeros(self.k, dtype=np.int64)
        self.total_pull_counts: IntArray = np.zeros(self.k, dtype=np.int64)
        self.total_reward_sums: FloatArray = np.zeros(self.k, dtype=np.float64)
        self._arm_cursor: int = 0
        self.reset()

    @property
    def name(self) -> str:
        return "Phased Elimination (Liu et al. 2025)"

    def reset(self) -> None:
        self.active_arms = np.arange(self.k, dtype=np.int64)
        self.phase_idx = 0
        self.phase_remaining_pulls = np.zeros(self.k, dtype=np.int64)
        self.total_pull_counts = np.zeros(self.k, dtype=np.int64)
        self.total_reward_sums = np.zeros(self.k, dtype=np.float64)
        self._arm_cursor = 0
        self._start_new_phase()

    def _phase_epsilon(self, phase_idx: int) -> float:
        return self.eps_init * (2.0 ** (-(phase_idx - 1)))

    def _start_new_phase(self) -> None:
        self.phase_idx += 1
        self.phase_remaining_pulls.fill(0)
        self._arm_cursor = 0

        if self.active_arms.size <= 1:
            return

        eps_l = self._phase_epsilon(self.phase_idx)
        active_X = self.features[self.active_arms]
        design_res = compute_g_optimal_design(active_X, tol=1e-4)
        pi_l = design_res.pi
        r_eff = float(design_res.effective_rank)

        k_act = float(self.active_arms.size)
        log_factor = float(
            np.log(
                (k_act * float(self.phase_idx) * float(self.phase_idx + 1))
                / self.delta
            )
        )
        raw_budget = (
            self.phase_scale
            * 2.0
            * r_eff
            * (self.sigma**2)
            * log_factor
            / (eps_l**2)
        )
        base_budget = max(3.5 * r_eff, raw_budget)

        for local_idx, global_arm in enumerate(self.active_arms):
            weight = float(pi_l[local_idx])
            if weight >= 0.01:
                target_i = max(1, int(np.round(weight * base_budget)))
                already_i = int(self.total_pull_counts[int(global_arm)])
                m_i = max(1, target_i - already_i)
                self.phase_remaining_pulls[int(global_arm)] = m_i

    def _finish_phase_and_eliminate(self) -> None:
        if self.active_arms.size <= 1:
            return

        active_counts = self.total_pull_counts[self.active_arms]
        pulled_local = np.where(active_counts > 0)[0]
        if pulled_local.size == 0:
            return

        pulled_arms = self.active_arms[pulled_local]
        active_X = self.features[self.active_arms]
        basis_idx, _ = select_well_conditioned_basis(active_X, tol=1e-8)
        r_eff = max(1, min(int(basis_idx.size), self.d))

        # Orthonormal basis Q_sub in R^{d x r_eff} of span(A_l)
        Q_full, _, _ = qr(active_X.T, mode="economic", pivoting=True)
        Q_sub: FloatArray = Q_full[:, :r_eff]

        Z_pulled: FloatArray = self.features[pulled_arms] @ Q_sub
        counts_pulled: FloatArray = self.total_pull_counts[pulled_arms].astype(
            np.float64
        )
        sums_pulled: FloatArray = self.total_reward_sums[pulled_arms]

        G_l: FloatArray = (
            1e-4 * np.eye(r_eff, dtype=np.float64)
            + (Z_pulled.T * counts_pulled) @ Z_pulled
        )
        b_sub: FloatArray = Z_pulled.T @ sums_pulled
        phi_hat_l: FloatArray = np.linalg.solve(G_l, b_sub)

        Z_active: FloatArray = active_X @ Q_sub
        est_means: FloatArray = Z_active @ phi_hat_l
        max_est_mean = float(np.max(est_means))

        eps_l = self._phase_epsilon(self.phase_idx)
        keep_mask = (max_est_mean - est_means) <= (2.0 * eps_l)
        if np.any(keep_mask):
            self.active_arms = self.active_arms[keep_mask]

    def select_arm(self) -> int:
        if self.active_arms.size == 1:
            return int(self.active_arms[0])

        while (
            self._arm_cursor < self.k
            and self.phase_remaining_pulls[self._arm_cursor] <= 0
        ):
            self._arm_cursor += 1

        if self._arm_cursor >= self.k:
            self._finish_phase_and_eliminate()
            self._start_new_phase()
            if self.active_arms.size == 1:
                return int(self.active_arms[0])
            while (
                self._arm_cursor < self.k
                and self.phase_remaining_pulls[self._arm_cursor] <= 0
            ):
                self._arm_cursor += 1

        return int(self._arm_cursor)

    def update(self, arm_idx: int, reward: float) -> None:
        self.total_pull_counts[arm_idx] += 1
        self.total_reward_sums[arm_idx] += float(reward)
        if self.phase_remaining_pulls[arm_idx] > 0:
            self.phase_remaining_pulls[arm_idx] -= 1