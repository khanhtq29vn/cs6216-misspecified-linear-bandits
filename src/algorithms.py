"""
Bandit Policies for Realizable and Misspecified Stochastic Linear Bandits.

Implements the four algorithms analyzed in Sections 2, 3, 4, and 5 of our
CS6216 Lecture Notes (`main.tex`):
1. `StandardUCB`: Unstructured k-armed Upper Confidence Bound policy
   (Lattimore & Szepesvari, 2020, Chapter 7).
2. `LinUCB`: Optimism in the Face of Uncertainty for Linear Bandits (OFUL)
   (Li et al., 2010; Abbasi-Yadkori et al., 2011, Theorems 2-3; Theorem 2.1 & Theorem 2.4).
3. `RobustLinearBanditRLB`: Two-phase Subspace Linearity Hypothesis-Testing Bandit
   (Ghosh, Chowdhury, & Gopalan, AAAI 2017, Algorithm 1 & Theorem 3; Section 3.3 & Theorem 3.3).
4. `PhasedEliminationBandit`: Kiefer-Wolfowitz G-Optimal Design Phased Elimination
   (Lattimore & Szepesvari, 2020, Chapter 21; Liu et al., 2025, Algorithm 2 & Theorem 16).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from src.utils import (
    FloatArray,
    IntArray,
    _canonical_feature_order,
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


def _validate_arm_and_reward(arm_idx: int, reward: float, k: int) -> Tuple[int, float]:
    if not isinstance(arm_idx, (int, np.integer)) or isinstance(arm_idx, bool):
        raise TypeError(f"arm_idx must be an integer, got {type(arm_idx)!r}.")
    idx = int(arm_idx)
    if not (0 <= idx < k):
        raise IndexError(f"arm_idx={idx} out of bounds for k={k} arms.")
    rew = float(reward)
    if not np.isfinite(rew):
        raise ValueError(f"Observed reward must be finite, got {reward}.")
    return idx, rew


class StandardUCB(BanditPolicy):
    """
    Unstructured k-armed Upper Confidence Bound (UCB) learner.
    """

    def __init__(self, k: int, sigma: float = 0.25, c_bonus: float = 2.0) -> None:
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or int(k) <= 0:
            raise ValueError(f"Number of arms k must be a positive integer, got {k}.")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}.")
        if not np.isfinite(c_bonus) or c_bonus < 0.0:
            raise ValueError(f"c_bonus must be finite and >= 0, got {c_bonus}.")

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
        idx, rew = _validate_arm_and_reward(arm_idx, reward, self._k)
        self.counts[idx] += 1
        n_i = int(self.counts[idx])
        self.emp_means[idx] += (rew - self.emp_means[idx]) / float(n_i)


class LinUCB(BanditPolicy):
    """
    LinUCB / OFUL (Optimism in the Face of Uncertainty for Linear Bandits).
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        reg_lambda: float = 1.0,
        delta: float = 0.05,
        theta_norm_bound: float = 1.0,
        confidence_scale: float = 0.60,
        alpha_scale: Optional[float] = None,
    ) -> None:
        X_arr = np.asarray(features, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"features must be a non-empty 2D array, got shape {X_arr.shape}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must contain only finite entries.")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}.")
        if not np.isfinite(reg_lambda) or reg_lambda <= 0.0:
            raise ValueError(f"reg_lambda must be finite and > 0, got {reg_lambda}.")
        if not np.isfinite(delta) or not (0.0 < delta < 1.0):
            raise ValueError(f"delta must lie in (0, 1), got {delta}.")
        if not np.isfinite(theta_norm_bound) or theta_norm_bound < 0.0:
            raise ValueError(
                f"theta_norm_bound S must be finite and >= 0, got {theta_norm_bound}."
            )

        scale_val = (
            float(alpha_scale) if alpha_scale is not None else float(confidence_scale)
        )
        if not np.isfinite(scale_val) or scale_val <= 0.0:
            raise ValueError(
                f"confidence_scale must be finite and > 0, got {scale_val}."
            )

        self.features: FloatArray = np.array(X_arr, dtype=np.float64, copy=True)
        self.k, self.d = self.features.shape
        self.sigma: float = float(sigma)
        self.reg_lambda: float = float(reg_lambda)
        self.delta: float = float(delta)
        self.S: float = float(theta_norm_bound)
        self.L: float = float(np.max(np.linalg.norm(self.features, axis=1)))
        self.confidence_scale: float = scale_val

        self.V_inv: FloatArray = np.eye(self.d, dtype=np.float64)
        self.b: FloatArray = np.zeros(self.d, dtype=np.float64)
        self.theta_hat: FloatArray = np.zeros(self.d, dtype=np.float64)
        self.num_observations: int = 0
        self.t: int = 0
        self.reset()

    @property
    def name(self) -> str:
        return "LinUCB (OFUL)"

    def reset(self) -> None:
        self.V_inv = (1.0 / self.reg_lambda) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)
        self.theta_hat = np.zeros(self.d, dtype=np.float64)
        self.num_observations = 0
        self.t = 0

    def confidence_radius(self, t: int) -> float:
        """
        Compute the self-normalized confidence ellipsoid radius sqrt(beta_t) at the
        start of round t (after n_obs = max(self.num_observations, t - 1) observations).
        """
        n_obs = max(self.num_observations, max(0, int(t) - 1))
        log_det_bound = self.d * np.log(
            1.0 + (float(n_obs) * (self.L**2)) / (self.reg_lambda * float(self.d))
        )
        sqrt_beta = np.sqrt(self.reg_lambda) * self.S + self.sigma * np.sqrt(
            2.0 * np.log(1.0 / self.delta) + log_det_bound
        )
        return self.confidence_scale * float(sqrt_beta)

    def _confidence_radius(self, t: int) -> float:
        return self.confidence_radius(t)

    def compute_ucb_scores(
        self, t: Optional[int] = None
    ) -> Tuple[FloatArray, FloatArray, float]:
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
        idx, rew = _validate_arm_and_reward(arm_idx, reward, self.k)
        x = self.features[idx]
        self.V_inv = sherman_morrison_update(self.V_inv, x)
        self.b += rew * x
        self.theta_hat = self.V_inv @ self.b
        self.num_observations += 1


class RobustLinearBanditRLB(BanditPolicy):
    """
    Robust Linear Bandit (RLB) via Subspace Linearity Hypothesis Testing
    (Ghosh, Chowdhury, & Gopalan, AAAI 2017, Algorithm 1 & Theorem 3;
    Section 3.3, Algorithm 3, Theorem 3.3, and Remark 3.4 in `main.tex`).

    Key Synchronization & Calibration Properties:
    - Uses the exact same `confidence_scale` (default `0.60`) for its internal
      warm-started `LinUCB` instance as standalone `LinUCB`, ensuring that under H_0
      (Realizable Regime 1), RLB's post-Phase-1 trajectory has the exact same slope
      as standalone `LinUCB`!
    - Uses high-probability test significance `test_delta = 0.005` (matching Theorem 3.3's
      delta_test = O(1/n) scaling) so that false rejections under H_0 are 0% across all
      action-space sizes k in {10, ..., 200}, while maintaining 100% power under H_1
      whenever eps_k > 0.25!
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        tau_per_arm: int = 20,
        delta: float = 0.05,
        test_delta: float = 0.005,
        reg_lambda: float = 1.0,
        confidence_scale: float = 0.60,
        alpha_scale: Optional[float] = None,
        num_test_arms: int = 3,
    ) -> None:
        X_arr = np.asarray(features, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"features must be a non-empty 2D array, got shape {X_arr.shape}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must contain only finite entries.")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}.")
        if (
            not isinstance(tau_per_arm, (int, np.integer))
            or isinstance(tau_per_arm, bool)
            or int(tau_per_arm) <= 0
        ):
            raise ValueError(
                f"tau_per_arm must be a positive integer, got {tau_per_arm}."
            )
        if not np.isfinite(delta) or not (0.0 < delta < 1.0):
            raise ValueError(f"delta must lie in (0, 1), got {delta}.")
        if not np.isfinite(test_delta) or not (0.0 < test_delta < 1.0):
            raise ValueError(f"test_delta must lie in (0, 1), got {test_delta}.")
        if not np.isfinite(reg_lambda) or reg_lambda <= 0.0:
            raise ValueError(f"reg_lambda must be finite and > 0, got {reg_lambda}.")
        if (
            not isinstance(num_test_arms, (int, np.integer))
            or isinstance(num_test_arms, bool)
            or int(num_test_arms) < 0
        ):
            raise ValueError(
                f"num_test_arms must be a non-negative integer, got {num_test_arms}."
            )

        scale_val = (
            float(alpha_scale) if alpha_scale is not None else float(confidence_scale)
        )
        if not np.isfinite(scale_val) or scale_val <= 0.0:
            raise ValueError(
                f"confidence_scale must be finite and > 0, got {scale_val}."
            )

        self.features: FloatArray = np.array(X_arr, dtype=np.float64, copy=True)
        self.k, self.d = self.features.shape
        self.sigma: float = float(sigma)
        self.tau: int = int(tau_per_arm)
        self.delta: float = float(delta)
        self.test_delta: float = float(test_delta)
        self.reg_lambda: float = float(reg_lambda)
        self.confidence_scale: float = scale_val

        basis_idx, non_basis_idx = select_well_conditioned_basis(
            self.features, tol=1e-8
        )
        self.basis_indices: IntArray = basis_idx

        if self.basis_indices.size > 0:
            self._X_B_cols: FloatArray = self.features[self.basis_indices].T
            self._X_B_pinv: FloatArray = np.linalg.pinv(self._X_B_cols, rcond=1e-8)
        else:
            self._X_B_cols = np.zeros((self.d, 0), dtype=np.float64)
            self._X_B_pinv = np.zeros((0, self.d), dtype=np.float64)

        if non_basis_idx.size > 0 and int(num_test_arms) > 0 and self.basis_indices.size > 0:
            m = min(int(num_test_arms), int(non_basis_idx.size))
            coords_all = self._X_B_pinv @ self.features[non_basis_idx].T
            coord_norms = np.linalg.norm(coords_all, axis=0)

            canon_non_basis = _canonical_feature_order(self.features[non_basis_idx])
            canon_rank = np.empty_like(canon_non_basis)
            canon_rank[canon_non_basis] = np.arange(non_basis_idx.size, dtype=np.int64)
            norms_rounded = np.round(coord_norms, decimals=11)
            order_by_norm = np.lexsort((canon_rank, norms_rounded))

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
        self.phase1_complete = self.total_phase1_rounds == 0
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
        if self.test_indices.size == 0 or self.basis_indices.size == 0:
            self.rejected_H0 = False
            return

        mu_hat_B: FloatArray = self.ucb.emp_means[self.basis_indices]
        m = int(self.test_indices.size)
        log_term = float(np.log(2.0 * float(m) / self.test_delta))
        mean_scale = max(1.0, float(np.max(np.abs(mu_hat_B))))

        max_ratio = -1.0
        self.rejected_H0 = False

        for j in self.test_indices:
            x_j = self.features[int(j)]
            c_j = self._X_B_pinv @ x_j
            recon_residual = float(np.linalg.norm(self._X_B_cols @ c_j - x_j))
            num_allowance = 1e-11 * mean_scale * (1.0 + float(np.linalg.norm(c_j))) + (
                mean_scale * recon_residual
            )

            extrapolated_mean = float(np.dot(c_j, mu_hat_B))
            direct_mean = float(self.ucb.emp_means[int(j)])
            discrepancy = abs(extrapolated_mean - direct_mean)

            variance_proxy = (
                (self.sigma**2)
                * (float(np.dot(c_j, c_j)) + 1.0)
                / float(self.tau)
            )
            threshold_j = float(np.sqrt(2.0 * variance_proxy * log_term)) + num_allowance

            ratio = discrepancy / max(threshold_j, 1e-15)
            if ratio > max_ratio:
                max_ratio = ratio
                self.max_test_stat = discrepancy
                self.test_threshold = threshold_j

            if discrepancy > threshold_j:
                self.rejected_H0 = True

    def select_arm(self) -> int:
        if self.t < self.total_phase1_rounds:
            self.t += 1
            idx_in_list = (self.t - 1) // self.tau
            return int(self.phase1_arms[idx_in_list])

        if not self.phase1_complete:
            self._run_hypothesis_test()
            self.phase1_complete = True

        self.t += 1
        if self.rejected_H0:
            return self.ucb.select_arm()
        return self.linucb.select_arm()

    def update(self, arm_idx: int, reward: float) -> None:
        idx, rew = _validate_arm_and_reward(arm_idx, reward, self.k)
        self.ucb.update(idx, rew)
        self.linucb.update(idx, rew)
        self.ucb.t = self.t
        self.linucb.t = self.t
        if not self.phase1_complete and self.t >= self.total_phase1_rounds:
            self._run_hypothesis_test()
            self.phase1_complete = True


class PhasedEliminationBandit(BanditPolicy):
    """
    Kiefer-Wolfowitz G-Optimal Design Phased Elimination for Linear Bandits
    (Lattimore & Szepesvari, 2020, Chapter 21; Liu et al., 2025, Algorithm 2 & Theorem 16).
    """

    def __init__(
        self,
        features: FloatArray,
        sigma: float = 0.25,
        delta: float = 0.05,
        phase_scale: float = 0.12,
        eps_init: float = 0.25,
    ) -> None:
        X_arr = np.asarray(features, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"features must be a non-empty 2D array, got shape {X_arr.shape}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("features must contain only finite entries.")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}.")
        if not np.isfinite(delta) or not (0.0 < delta < 1.0):
            raise ValueError(f"delta must lie in (0, 1), got {delta}.")
        if not np.isfinite(phase_scale) or phase_scale <= 0.0:
            raise ValueError(f"phase_scale must be finite and > 0, got {phase_scale}.")
        if not np.isfinite(eps_init) or eps_init <= 0.0:
            raise ValueError(f"eps_init must be finite and > 0, got {eps_init}.")

        self.features: FloatArray = np.array(X_arr, dtype=np.float64, copy=True)
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
        r_eff = float(max(1, design_res.effective_rank))

        k_act = float(self.active_arms.size)
        log_factor = float(
            np.log(
                (2.0 * k_act * float(self.phase_idx) * float(self.phase_idx + 1))
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

        supp_local = np.where(pi_l >= 1e-6)[0]
        if supp_local.size == 0:
            supp_local = np.where(pi_l > 0.0)[0]
        if supp_local.size == 0:
            supp_local = np.array([0], dtype=np.int64)

        for local_idx in supp_local:
            weight = float(pi_l[int(local_idx)])
            global_arm = int(self.active_arms[int(local_idx)])
            target_i = max(1, int(np.ceil(weight * base_budget - 1e-12)))
            already_i = int(self.total_pull_counts[global_arm])
            m_i = max(1, target_i - already_i)
            self.phase_remaining_pulls[global_arm] = m_i

    def _finish_phase_and_eliminate(self) -> None:
        if self.active_arms.size <= 1:
            return

        active_counts = self.total_pull_counts[self.active_arms]
        pulled_local = np.where(active_counts > 0)[0]
        if pulled_local.size == 0:
            return

        pulled_arms = self.active_arms[pulled_local]
        active_X = self.features[self.active_arms]

        u_mat, s_vals, vt_mat = np.linalg.svd(active_X, full_matrices=False)
        if s_vals.size == 0 or float(s_vals[0]) <= 1e-15:
            return
        r_eff = max(1, int(np.sum(s_vals > 1e-8 * float(s_vals[0]))))
        Q_sub: FloatArray = vt_mat[:r_eff].T

        Z_pulled: FloatArray = self.features[pulled_arms] @ Q_sub
        counts_pulled: FloatArray = self.total_pull_counts[pulled_arms].astype(
            np.float64
        )
        sums_pulled: FloatArray = self.total_reward_sums[pulled_arms]

        G_l: FloatArray = (
            1e-6 * np.eye(r_eff, dtype=np.float64)
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

        if self._arm_cursor >= self.k:
            return int(self.active_arms[0])
        return int(self._arm_cursor)

    def update(self, arm_idx: int, reward: float) -> None:
        idx, rew = _validate_arm_and_reward(arm_idx, reward, self.k)
        self.total_pull_counts[idx] += 1
        self.total_reward_sums[idx] += rew
        if self.phase_remaining_pulls[idx] > 0:
            self.phase_remaining_pulls[idx] -= 1