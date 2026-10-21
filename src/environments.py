"""
Stochastic Linear Bandit Environments under Exact Realizability and Model Misspecification.

Implements three benchmark environments aligned with our CS6216 lecture notes:
1. Exact Realizability (Li et al., 2010; Abbasi-Yadkori et al., 2011):
       f_0(x_i) = <x_i, theta^*>
2. Sparse Subspace Misspecification (Ghosh, Chowdhury, & Gopalan, AAAI 2017):
       f_0(x_i) = <x_i, theta^*> + eps_i
   Constructed following Theorem 1 of Ghosh et al. (2017) to demonstrate why unmodified
   OFUL/LinUCB suffers catastrophic linear regret Omega(n) when an optimal arm has a
   positive sparse deviation outside the linear extrapolation of basis arms.
3. Gap-Adjusted Misspecification (rho-GAM, Liu, Qiao, Yin, Bogunovic, & Wang, 2025):
       |f_{w^*}(x_i) - f_0(x_i)| <= rho * (f^* - f_0(x_i)),  for rho in [0, 1)
   Constructed following Definition 2 of Liu et al. (2025), where the uniform sup-norm
   misspecification ||eps||_inf is large on far-from-optimal arms, yet the relative
   gap-adjusted misspecification is bounded by rho.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class LinearBanditEnvironment:
    """
    Finite-armed stochastic linear bandit environment with optional misspecification.

    Attributes
    ----------
    name : str
        Descriptive name of the environment regime.
    features : np.ndarray of shape (k, d)
        Known feature vectors x_1, ..., x_k in R^d.
    theta_star : np.ndarray of shape (d,)
        Reference linear parameter vector theta^* (or w^* in rho-GAM).
    misspecification : np.ndarray of shape (k,)
        Additive deviation vector eps = (eps_1, ..., eps_k)^T where
        f_0(x_i) = <x_i, theta^*> + eps_i.
    sigma : float
        Standard deviation of the observation noise eta_t ~ N(0, sigma^2).
    """

    name: str
    features: np.ndarray
    theta_star: np.ndarray
    misspecification: np.ndarray
    sigma: float

    def __post_init__(self) -> None:
        self.k, self.d = self.features.shape
        self.linear_means = self.features @ self.theta_star
        self.true_means = self.linear_means + self.misspecification
        self.optimal_arm = int(np.argmax(self.true_means))
        self.optimal_mean = float(self.true_means[self.optimal_arm])
        self.suboptimality_gaps = self.optimal_mean - self.true_means

    def step(self, arm_idx: int, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Pull arm `arm_idx` and observe noisy reward Y_t = f_0(x_{A_t}) + eta_t.

        Returns
        -------
        reward : float
            Stochastic reward Y_t.
        regret : float
            Instantaneous pseudo-regret Delta_{A_t} = f_0(x^*) - f_0(x_{A_t}).
        """
        mean_reward = float(self.true_means[arm_idx])
        reward = mean_reward + float(rng.normal(0.0, self.sigma))
        inst_regret = float(self.suboptimality_gaps[arm_idx])
        return reward, inst_regret

    def diagnostics(self) -> Dict[str, float]:
        """
        Compute exact misspecification metrics (uniform l_inf norm and rho-GAM parameter).
        """
        eps_inf = float(np.max(np.abs(self.misspecification)))
        non_opt_mask = self.suboptimality_gaps > 1e-12
        if np.any(non_opt_mask):
            gam_ratios = (
                np.abs(self.linear_means[non_opt_mask] - self.true_means[non_opt_mask])
                / self.suboptimality_gaps[non_opt_mask]
            )
            rho_gam = float(np.max(gam_ratios))
        else:
            rho_gam = 0.0

        min_nonzero_gap = (
            float(np.min(self.suboptimality_gaps[non_opt_mask]))
            if np.any(non_opt_mask)
            else 0.0
        )
        return {
            "k": self.k,
            "d": self.d,
            "optimal_arm": self.optimal_arm,
            "optimal_mean": self.optimal_mean,
            "min_gap": min_nonzero_gap,
            "eps_linf": eps_inf,
            "rho_gam": rho_gam,
        }


def make_realizable_env(
    k: int = 40, d: int = 5, sigma: float = 0.30, seed: int = 42
) -> LinearBanditEnvironment:
    """
    Benchmark 1: Exact Realizable Linear Bandit (eps = 0).

    Constructs k arms in R^d with unit-bounded features ||x_i||_2 <= 1 and a clear
    suboptimality gap so that standard LinUCB/OFUL achieves O(d sqrt(n)) regret
    and significantly outperforms unstructured UCB O(sqrt(k n)).
    """
    rng = np.random.default_rng(seed)
    theta_star = np.array([0.80, 0.45, -0.35, 0.25, -0.20][:d], dtype=float)
    if len(theta_star) < d:
        extra = rng.uniform(-0.3, 0.3, size=d - len(theta_star))
        theta_star = np.concatenate([theta_star, extra])
    theta_star /= np.linalg.norm(theta_star)

    features = np.zeros((k, d), dtype=float)
    # Arm 0 is aligned with theta^* (unique global optimum with mean 1.0)
    features[0] = theta_star.copy()

    # Remaining k-1 arms are sampled on the unit sphere with inner product <= 0.82
    idx = 1
    while idx < k:
        cand = rng.normal(0.0, 1.0, size=d)
        cand /= np.linalg.norm(cand)
        if np.dot(cand, theta_star) <= 0.82:
            features[idx] = cand
            idx += 1

    misspecification = np.zeros(k, dtype=float)
    return LinearBanditEnvironment(
        name="Realizable Linear Bandit (eps = 0)",
        features=features,
        theta_star=theta_star,
        misspecification=misspecification,
        sigma=sigma,
    )


def make_ghosh_sparse_misspecified_env(
    k: int = 20,
    d: int = 5,
    sigma: float = 0.25,
    eps_sparse: float = 0.45,
    seed: int = 42,
) -> LinearBanditEnvironment:
    """
    Benchmark 2: Sparse Subspace Misspecification (Ghosh, Chowdhury, & Gopalan, 2017).

    Constructs the exact failure geometry analyzed in Theorem 1 of Ghosh et al. (2017):
    - Arms 0, ..., d-1 form the canonical orthonormal basis e_1, ..., e_d with zero
      misspecification (eps_i = 0) and linear rewards:
          theta^* = (0.70, 0.45, 0.35, ..., 0.25)^T.
      Among all linear predictions, Arm 0 (e_1) has the highest linear reward (0.70).
    - Arms d, ..., k-2 lie in the interior of the unit ball with zero misspecification
      and linear rewards strictly below 0.60.
    - Arm k-1 (the corrupted optimal arm) has feature vector:
          x_{k-1} = 0.50 * e_1 + 0.30 * e_2   (||x_{k-1}||_2 = 0.583 < 1).
      Its linearly extrapolated reward is <x_{k-1}, theta^*> = 0.50(0.70) + 0.30(0.45) = 0.485.
      However, it carries a sparse positive deviation eps_{k-1} = +0.45, making its true
      mean reward f_0(x_{k-1}) = 0.935 > 0.70 (the unique global optimum!).

    Why Unmodified LinUCB Fails (Theorem 1):
    Because x_{k-1} lies in span(e_1, e_2) with coordinates (0.50, 0.30) whose sum is
    0.80 < 1, pulling Arm 0 (e_1) shrinks the confidence ellipsoid along e_1 and drags
    the linear UCB index of Arm k-1 permanently below 0.70. Unmodified LinUCB locks onto
    suboptimal Arm 0 and suffers linear regret Delta_0 * n = 0.235 * n!
    """
    rng = np.random.default_rng(seed)
    theta_star = np.linspace(0.45, 0.25, d)
    theta_star[0] = 0.70
    if d >= 2:
        theta_star[1] = 0.45

    features = np.zeros((k, d), dtype=float)
    # First d arms: canonical basis e_1, ..., e_d
    for i in range(d):
        features[i, i] = 1.0

    # Intermediate arms d .. k-2: uncorrupted arms with moderate linear rewards
    for i in range(d, k - 1):
        weights = rng.uniform(0.1, 0.6, size=d)
        weights /= np.linalg.norm(weights) * 1.15
        features[i] = weights

    # Last arm k-1: lies in span(e_1, e_2) with shrinkage factor < 1
    features[k - 1] = 0.0
    features[k - 1, 0] = 0.50
    if d >= 2:
        features[k - 1, 1] = 0.30

    misspecification = np.zeros(k, dtype=float)
    misspecification[k - 1] = eps_sparse

    return LinearBanditEnvironment(
        name=f"Sparse Misspecified Linear Bandit (eps_k = {eps_sparse:.2f})",
        features=features,
        theta_star=theta_star,
        misspecification=misspecification,
        sigma=sigma,
    )


def make_gap_adjusted_env(
    k: int = 40,
    d: int = 5,
    sigma: float = 0.25,
    rho: float = 0.30,
    skew_suboptimal_bias: bool = False,
    seed: int = 42,
) -> LinearBanditEnvironment:
    """
    Benchmark 3: rho-Gap-Adjusted Misspecification (rho-GAM, Liu, Bogunovic et al., 2025).

    Constructs a globally nonlinear reward function f_0(x) whose best linear approximation
    f_{w^*}(x) = <x, w^*> satisfies Definition 2 of Liu et al. (2025) with exact parameter
    rho in [0, 1):
        1) max_{x in X} f_{w^*}(x) = f_{w^*}(x^*) = f_0(x^*) = f^* = 1.0  (eps(x^*) = 0)
        2) For all x != x^*:
               |f_{w^*}(x) - f_0(x)| / (f^* - f_0(x)) = rho.

    Algebraic Construction:
    For each suboptimal arm i >= 1 with linear gap Delta_i^{lin} = f^* - <x_i, w^*> > 0
    and sign s_i in {-1, +1}, we set the true gap to:
        Delta_i = Delta_i^{lin} / (1 + s_i * rho),   f_0(x_i) = f^* - Delta_i.
    Then:
        eps_i = f_0(x_i) - <x_i, w^*> = Delta_i^{lin} - Delta_i = s_i * rho * Delta_i,
    so |eps_i| / Delta_i = rho holds identically for every suboptimal arm, while the
    uniform sup-norm error ||eps||_inf = max_i (rho * Delta_i) is large on arms far
    from x^*!
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must lie in [0, 1), got {rho}.")

    rng = np.random.default_rng(seed)
    w_star = np.zeros(d, dtype=float)
    w_star[0] = 1.0  # Clean reference direction w^* = e_1 (unit norm, f^* = 1.0)

    features = np.zeros((k, d), dtype=float)
    # Arm 0 is the unique optimal arm x^* = w^* with f_0(x^*) = <x^*, w^*> = 1.0
    features[0] = w_star.copy()

    # Arms 1 .. d-1 complete a full-rank basis with large suboptimality gaps
    for j in range(1, d):
        vec = np.zeros(d, dtype=float)
        vec[0] = -0.35
        vec[j] = np.sqrt(1.0 - 0.35**2)
        features[j] = vec

    # Remaining arms d .. k-1 cover a spectrum of linear values in [-0.50, 0.84]
    c1_vals = np.linspace(0.84, -0.50, k - d)
    for idx_offset, c1 in enumerate(c1_vals):
        arm_idx = d + idx_offset
        perp = rng.normal(0.0, 1.0, size=d - 1)
        perp /= np.linalg.norm(perp)
        radius = np.sqrt(max(0.0, 1.0 - c1**2))
        features[arm_idx, 0] = c1
        features[arm_idx, 1:] = radius * perp

    linear_means = features @ w_star
    f_star = float(linear_means[0])  # 1.0
    linear_gaps = f_star - linear_means

    misspecification = np.zeros(k, dtype=float)
    for i in range(1, k):
        if skew_suboptimal_bias:
            # Coherent perturbation sign correlated with the second coordinate x_{i,1}
            # This creates systematic bias in adaptive ridge regression (LinUCB) when
            # rho is large, highlighting the robustness advantage of G-optimal Phased Elimination!
            s_i = 1.0 if features[i, 1] >= 0.0 else -1.0
        else:
            s_i = 1.0 if (i % 2 == 1) else -1.0

        true_gap_i = linear_gaps[i] / (1.0 + s_i * rho)
        true_mean_i = f_star - true_gap_i
        misspecification[i] = true_mean_i - linear_means[i]

    return LinearBanditEnvironment(
        name=f"Gap-Adjusted Misspecified Bandit (rho = {rho:.2f})",
        features=features,
        theta_star=w_star,
        misspecification=misspecification,
        sigma=sigma,
    )