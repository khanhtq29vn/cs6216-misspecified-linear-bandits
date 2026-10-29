"""
Stochastic Linear Bandit Environments across Three Model Misspecification Regimes.

Strictly implements and mathematically verifies the three regimes studied in
Sections 2, 3, 4, and 5 of our CS6216 Lecture Notes (`main.tex`):

1. Regime 1 — Exact Linear Realizability (Section 2; Li et al., 2010; Abbasi-Yadkori et al., 2011):
       f_0(x_i) = <x_i, theta^*>,   eps_i = 0  for all i in {0, ..., k-1}.

2. Regime 2 — Sparse Subspace Misspecification (Section 3.2; Ghosh, Chowdhury, & Gopalan, AAAI 2017):
       f_0(x_i) = <x_i, theta^*> + eps_i.
   Constructed according to Theorem 1 of Ghosh et al. (2017): d arms form the canonical
   orthonormal basis e_1, ..., e_d with zero error (eps_i = 0) and ordered linear means
   theta_1^* > theta_2^* >= ... >= theta_d^* > 0. A corrupted arm x_{corr} lies in
   span(e_1, e_2) with shrinkage coordinates (alpha_1 + alpha_2 < 1) so its linearly
   extrapolated reward is mu_{corr}^{lin} < theta_1^*, but carries a positive sparse
   deviation eps_k > theta_1^* - mu_{corr}^{lin}, making it the unique true optimal arm.
   Importantly, arm indices are randomly permuted so no algorithm can exploit array ordering.

3. Regime 3 — rho-Gap-Adjusted Misspecification (Section 4; Liu, Qiao, Yin, Bogunovic, & Wang, 2025):
       max_{x in X} f_{w^*}(x) = f^*   and   |f_{w^*}(x_i) - f_0(x_i)| <= rho * (f^* - f_0(x_i))
   for rho in [0, 1). Automatically verifies Proposition 4.2 (Multiplicative Gap Sandwiching
   and Exact Optimality Preservation) upon initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np

from src.utils import FloatArray


@dataclass(frozen=True)
class EnvironmentDiagnostics:
    """
    Exact structural and misspecification metrics of a `LinearBanditEnvironment`.
    Supports both attribute access (`diag.eps_linf_norm`) and dictionary-style
    lookup (`diag["eps_linf"]` or `diag["eps_linf_norm"]`).
    """

    regime_name: str
    k: int
    d: int
    optimal_arm: int
    optimal_mean: float
    second_best_mean: float
    min_suboptimality_gap: float
    max_feature_norm: float
    parameter_norm: float
    eps_linf_norm: float
    rho_gam_ratio: float
    preserves_linear_optimum: bool

    def __getitem__(self, key: str) -> Any:
        alias_map = {
            "eps_linf": "eps_linf_norm",
            "rho_gam": "rho_gam_ratio",
            "f_star": "optimal_mean",
            "delta_min": "min_suboptimality_gap",
        }
        attr_name = alias_map.get(key, key)
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        raise KeyError(f"Unknown diagnostic key: {key!r}")


class LinearBanditEnvironment:
    """
    Immutable, self-verifying finite-armed stochastic linear bandit environment.

    Reward generation model at round t when arm i in {0, ..., k-1} is pulled:
        Y_t = f_0(x_i) + eta_t = (<x_i, theta^*> + eps_i) + eta_t,
    where eta_t ~ N(0, sigma^2) is conditional sigma-sub-Gaussian noise.
    """

    def __init__(
        self,
        name: str,
        features: FloatArray,
        theta_star: FloatArray,
        misspecification: FloatArray,
        sigma: float,
        feature_norm_bound: float = 1.0,
        expected_rho_gam: Optional[float] = None,
        corrupted_arm_idx: Optional[int] = None,
        basis_arm_1_idx: Optional[int] = None,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {features.shape}.")
        k, d = features.shape
        if theta_star.shape != (d,):
            raise ValueError(
                f"theta_star shape {theta_star.shape} does not match d={d}."
            )
        if misspecification.shape != (k,):
            raise ValueError(
                f"misspecification shape {misspecification.shape} does not match k={k}."
            )
        if sigma < 0.0:
            raise ValueError(f"Noise standard deviation sigma must be >= 0, got {sigma}.")

        self._name: str = name
        self._k: int = k
        self._d: int = d
        self._sigma: float = float(sigma)
        self._L: float = float(feature_norm_bound)

        # Store read-only copies of mathematical arrays to guarantee immutability
        self._features: FloatArray = np.array(features, dtype=np.float64, copy=True)
        self._theta_star: FloatArray = np.array(theta_star, dtype=np.float64, copy=True)
        self._misspecification: FloatArray = np.array(
            misspecification, dtype=np.float64, copy=True
        )
        self._features.setflags(write=False)
        self._theta_star.setflags(write=False)
        self._misspecification.setflags(write=False)

        self._linear_means: FloatArray = self._features @ self._theta_star
        self._true_means: FloatArray = self._linear_means + self._misspecification
        self._linear_means.setflags(write=False)
        self._true_means.setflags(write=False)

        self._optimal_arm: int = int(np.argmax(self._true_means))
        self._optimal_mean: float = float(self._true_means[self._optimal_arm])
        self._suboptimality_gaps: FloatArray = self._optimal_mean - self._true_means
        self._suboptimality_gaps.setflags(write=False)

        self._corrupted_arm_idx: Optional[int] = corrupted_arm_idx
        self._basis_arm_1_idx: Optional[int] = basis_arm_1_idx

        # Self-verification of Assumption 1.1 and Proposition 4.2 (if rho-GAM)
        self._verify_invariants(expected_rho_gam)

    def _verify_invariants(self, expected_rho_gam: Optional[float]) -> None:
        max_norm = float(np.max(np.linalg.norm(self._features, axis=1)))
        if max_norm > self._L + 1e-7:
            raise AssertionError(
                f"Feature norm bound violated: max_i ||x_i||_2 = {max_norm:.6f} > L = {self._L}."
            )

        if expected_rho_gam is not None:
            # Verify Definition 4.1 (rho-GAM) and Proposition 4.2 (Gap Sandwiching & Optimality Preservation)
            f_star = self._optimal_mean
            lin_max = float(np.max(self._linear_means))
            if abs(lin_max - f_star) > 1e-7:
                raise AssertionError(
                    f"rho-GAM peak condition violated: max_x f_{{w^*}}(x) = {lin_max} != f^* = {f_star}."
                )

            non_opt = self._suboptimality_gaps > 1e-10
            if np.any(non_opt):
                true_gaps = self._suboptimality_gaps[non_opt]
                lin_gaps = f_star - self._linear_means[non_opt]
                ratios = np.abs(self._misspecification[non_opt]) / true_gaps
                max_ratio = float(np.max(ratios))
                if max_ratio > expected_rho_gam + 1e-7:
                    raise AssertionError(
                        f"rho-GAM ratio violated: max |eps_i|/Delta_i = {max_ratio:.6f} > rho = {expected_rho_gam}."
                    )
                # Verify Proposition 4.2(i): (1 - rho) Delta(x) <= f^* - f_{w^*}(x) <= (1 + rho) Delta(x)
                lower_ok = np.all(
                    lin_gaps >= (1.0 - expected_rho_gam) * true_gaps - 1e-7
                )
                upper_ok = np.all(
                    lin_gaps <= (1.0 + expected_rho_gam) * true_gaps + 1e-7
                )
                if not (lower_ok and upper_ok):
                    raise AssertionError(
                        "Proposition 4.2 multiplicative gap-sandwiching inequality failed verification."
                    )

    @property
    def name(self) -> str:
        return self._name

    @property
    def k(self) -> int:
        return self._k

    @property
    def d(self) -> int:
        return self._d

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def features(self) -> FloatArray:
        return self._features

    @property
    def theta_star(self) -> FloatArray:
        return self._theta_star

    @property
    def misspecification(self) -> FloatArray:
        return self._misspecification

    @property
    def linear_means(self) -> FloatArray:
        return self._linear_means

    @property
    def true_means(self) -> FloatArray:
        return self._true_means

    @property
    def optimal_arm(self) -> int:
        return self._optimal_arm

    @property
    def optimal_mean(self) -> float:
        return self._optimal_mean

    @property
    def suboptimality_gaps(self) -> FloatArray:
        return self._suboptimality_gaps

    @property
    def corrupted_arm_idx(self) -> Optional[int]:
        return self._corrupted_arm_idx

    @property
    def basis_arm_1_idx(self) -> Optional[int]:
        return self._basis_arm_1_idx

    def step(self, arm_idx: int, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Execute one sequential interaction step by pulling `arm_idx` in {0, ..., k-1}.

        Returns
        -------
        reward : float
            Noisy scalar observation Y_t = f_0(x_{arm_idx}) + eta_t.
        inst_regret : float
            Instantaneous pseudo-regret Delta(x_{arm_idx}) = f^* - f_0(x_{arm_idx}) >= 0.
        """
        if not (0 <= arm_idx < self._k):
            raise IndexError(f"arm_idx={arm_idx} out of bounds for k={self._k} arms.")
        reward = float(self._true_means[arm_idx]) + float(rng.normal(0.0, self._sigma))
        inst_regret = float(self._suboptimality_gaps[arm_idx])
        return reward, inst_regret

    def diagnostics(self) -> EnvironmentDiagnostics:
        """
        Compute comprehensive mathematical diagnostics of the environment.
        """
        eps_linf = float(np.max(np.abs(self._misspecification)))
        non_opt = self._suboptimality_gaps > 1e-10
        if np.any(non_opt):
            rho_gam = float(
                np.max(
                    np.abs(self._misspecification[non_opt])
                    / self._suboptimality_gaps[non_opt]
                )
            )
            min_gap = float(np.min(self._suboptimality_gaps[non_opt]))
            second_best = float(np.max(self._true_means[non_opt]))
        else:
            rho_gam = 0.0
            min_gap = 0.0
            second_best = self._optimal_mean

        lin_opt = int(np.argmax(self._linear_means))
        return EnvironmentDiagnostics(
            regime_name=self._name,
            k=self._k,
            d=self._d,
            optimal_arm=self._optimal_arm,
            optimal_mean=self._optimal_mean,
            second_best_mean=second_best,
            min_suboptimality_gap=min_gap,
            max_feature_norm=float(np.max(np.linalg.norm(self._features, axis=1))),
            parameter_norm=float(np.linalg.norm(self._theta_star)),
            eps_linf_norm=eps_linf,
            rho_gam_ratio=rho_gam,
            preserves_linear_optimum=(lin_opt == self._optimal_arm),
        )

    def summary(self) -> str:
        diag = self.diagnostics()
        return (
            f"[{diag.regime_name}] k={diag.k}, d={diag.d} | "
            f"f*={diag.optimal_mean:.3f} (arm {diag.optimal_arm}), "
            f"Delta_min={diag.min_suboptimality_gap:.3f} | "
            f"||eps||_inf={diag.eps_linf_norm:.3f}, rho_GAM={diag.rho_gam_ratio:.3f}, "
            f"preserves_opt={diag.preserves_linear_optimum}"
        )


def make_realizable_env(
    k: int = 50,
    d: int = 5,
    sigma: float = 0.25,
    min_gap: float = 0.18,
    permute_arms: bool = True,
    seed: int = 42,
) -> LinearBanditEnvironment:
    """
    Construct Regime 1: Exact Linear Realizability (eps = 0, Section 2 & Section 5.1).

    Parameters
    ----------
    k : int
        Number of actions in X subset R^d.
    d : int
        Feature dimension.
    sigma : float
        Sub-Gaussian noise scale.
    min_gap : float
        Minimum suboptimality gap Delta_min = 1.0 - max_{i != i^*} <x_i, theta^*>.
    permute_arms : bool
        If True, randomly permutes arm indices so the optimal arm is not at index 0.
    seed : int
        Random seed for reproducible generation and arm index permutation.
    """
    rng = np.random.default_rng(seed)
    base_theta = np.array([0.80, 0.45, -0.35, 0.25, -0.20], dtype=np.float64)
    if d <= len(base_theta):
        theta_star = base_theta[:d].copy()
    else:
        extra = rng.uniform(-0.3, 0.3, size=d - len(base_theta))
        theta_star = np.concatenate([base_theta, extra])
    theta_star /= float(np.linalg.norm(theta_star))

    features = np.zeros((k, d), dtype=np.float64)
    features[0] = theta_star.copy()  # Unique optimal arm with <x^*, theta^*> = 1.0

    # Suboptimal unit-norm arms with <x_i, theta^*> in [0.55, 1.0 - min_gap]
    # so suboptimality gaps lie in [min_gap, 0.45] while spanning R^d.
    max_subopt_mean = max(0.20, 1.0 - float(min_gap))
    min_subopt_mean = min(0.55, max_subopt_mean - 0.10)
    c1_vals = np.linspace(max_subopt_mean, min_subopt_mean, num=k - 1, dtype=np.float64)

    for idx in range(1, k):
        c1 = float(c1_vals[idx - 1])
        perp_radius = float(np.sqrt(max(0.0, 1.0 - c1**2)))
        while True:
            raw = rng.normal(0.0, 1.0, size=d)
            orth = raw - float(np.dot(raw, theta_star)) * theta_star
            norm_orth = float(np.linalg.norm(orth))
            if norm_orth > 1e-8:
                orth /= norm_orth
                break
        features[idx] = c1 * theta_star + perp_radius * orth

    if permute_arms:
        perm = rng.permutation(k)
        features = features[perm]

    misspecification = np.zeros(k, dtype=np.float64)
    return LinearBanditEnvironment(
        name=f"Regime 1: Exact Realizability (eps=0, k={k}, d={d})",
        features=features,
        theta_star=theta_star,
        misspecification=misspecification,
        sigma=sigma,
        expected_rho_gam=0.0,
    )


def make_ghosh_sparse_misspecified_env(
    k: int = 20,
    d: int = 5,
    sigma: float = 0.25,
    eps_sparse: float = 0.45,
    alpha_1: float = 0.50,
    alpha_2: float = 0.30,
    permute_arms: bool = True,
    seed: int = 42,
) -> LinearBanditEnvironment:
    """
    Construct Regime 2: Sparse Subspace Misspecification (Theorem 1 of Ghosh et al., AAAI 2017;
    Theorem 3.2 and Equations (11)-(12) in Section 5.1 of `main.tex`).

    Mathematical Structure:
    - Basis arms e_1, ..., e_d have unit norm ||e_i||_2 = 1.0, zero misspecification
      (eps_i = 0), and true means theta^* = (0.70, 0.45, 0.3833, 0.3167, 0.25)^T.
      Thus Basis Arm 1 (e_1) has the largest linear reward mu_1 = theta_1^* = 0.70.
    - Intermediate uncorrupted arms have Euclidean norm strictly in [0.35, 0.48]
      (< ||x_corr||_2 = sqrt(0.34) = 0.5831 < 1.0) and zero misspecification.
    - The corrupted arm x_corr lies in span(e_1, e_2) with shrinkage coordinates:
          x_corr = alpha_1 * e_1 + alpha_2 * e_2   (alpha_1 + alpha_2 = 0.80 < 1).
      Its linearly extrapolated reward is:
          mu_corr^{lin} = 0.50 * 0.70 + 0.30 * 0.45 = 0.485 < 0.70.
    - Phase-Transition Threshold (Equation 12 of `main.tex`):
          eps_k^* = theta_1^* - mu_corr^{lin} = 0.70 - 0.485 = 0.215.
      Whenever eps_sparse > 0.215, x_corr is the unique true optimal arm
      (at eps_sparse = 0.45, mu_corr = 0.935 and Delta_1 = 0.235), yet unmodified
      LinUCB/OFUL locks onto Basis Arm 1 and suffers linear regret Omega(Delta_1 * n).
    """
    if d < 2:
        raise ValueError(f"Ghosh counterexample requires d >= 2, got d={d}.")
    if k < d + 1:
        raise ValueError(f"k must be at least d + 1 = {d + 1}, got k={k}.")
    if alpha_1 + alpha_2 >= 1.0 or alpha_1 <= 0.0 or alpha_2 < 0.0:
        raise ValueError(
            "Shrinkage coordinates must satisfy alpha_1 > 0, alpha_2 >= 0, alpha_1 + alpha_2 < 1."
        )

    rng = np.random.default_rng(seed)
    theta_star = np.linspace(0.45, 0.25, d, dtype=np.float64)
    theta_star[0] = 0.70
    theta_star[1] = 0.45

    features = np.zeros((k, d), dtype=np.float64)
    for i in range(d):
        features[i, i] = 1.0

    for i in range(d, k - 1):
        vec = rng.uniform(0.10, 0.50, size=d)
        vec = (vec / float(np.linalg.norm(vec))) * float(rng.uniform(0.35, 0.48))
        features[i] = vec

    features[k - 1, 0] = alpha_1
    features[k - 1, 1] = alpha_2

    misspecification = np.zeros(k, dtype=np.float64)
    misspecification[k - 1] = float(eps_sparse)

    corrupted_idx = k - 1
    basis_1_idx = 0

    if permute_arms:
        perm = rng.permutation(k)
        features = features[perm]
        misspecification = misspecification[perm]
        corrupted_idx = int(np.where(perm == (k - 1))[0][0])
        basis_1_idx = int(np.where(perm == 0)[0][0])

    return LinearBanditEnvironment(
        name=f"Regime 2: Ghosh Sparse Misspec (eps_k={eps_sparse:.2f}, k={k}, d={d})",
        features=features,
        theta_star=theta_star,
        misspecification=misspecification,
        sigma=sigma,
        corrupted_arm_idx=corrupted_idx,
        basis_arm_1_idx=basis_1_idx,
    )


def make_gap_adjusted_env(
    k: int = 40,
    d: int = 5,
    sigma: float = 0.25,
    rho: float = 0.30,
    skew_suboptimal_bias: bool = False,
    permute_arms: bool = True,
    seed: int = 42,
) -> LinearBanditEnvironment:
    """
    Construct Regime 3: rho-Gap-Adjusted Misspecification (Definition 4.1 & Equation (13)
    in Section 5.1 of `main.tex`; Liu, Qiao, Yin, Bogunovic, & Wang, 2025).

    Exact Algebraic Construction:
    - Reference parameter w^* = e_1 in R^d (||w^*||_2 = 1), optimal action x^* = w^*
      with f_0(x^*) = <x^*, w^*> = f^* = 1.0 (so eps(x^*) = 0).
    - Far-from-optimal basis arm x_1 = c_1 e_1 + sqrt(1 - c_1^2) e_2 with c_1 = -31/60 = -0.5167
      (linear gap Delta_1^{lin} = 91/60) and diagnostic companion arm x_2 = -0.30 e_1 - sqrt(1 - 0.30^2) e_2.
      At rho = 0.30 with s_1 = +1, the global sup-norm misspecification is:
          ||eps||_inf = |eps_1| = 0.30 * (91/60) / (1 + 0.30) = 0.3500 (exact match to Section 5.1!).
    - Remaining basis arms span e_3, ..., e_d and grid arms span <x_i, w^*> in [-0.40, 0.74]
      within span(e_1, e_3, ..., e_d).
    - For every suboptimal arm i >= 1 with linear gap Delta_i^{lin} = f^* - <x_i, w^*> > 0
      and sign s_i in {-1, +1}, we set:
          Delta_i = Delta_i^{lin} / (1 + s_i * rho),     f_0(x_i) = f^* - Delta_i.
      Then:
          eps_i = f_0(x_i) - <x_i, w^*> = Delta_i^{lin} - Delta_i = s_i * rho * Delta_i,
      which guarantees |eps_i| / Delta_i = rho identically for all i != i^*.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho-GAM parameter rho must lie in [0, 1), got {rho}.")
    if k <= d:
        raise ValueError(f"k must exceed d={d}, got k={k}.")

    rng = np.random.default_rng(seed)
    w_star = np.zeros(d, dtype=np.float64)
    w_star[0] = 1.0

    features = np.zeros((k, d), dtype=np.float64)
    features[0] = w_star.copy()  # Unique optimal arm x^* = w^*

    # Arm 1: Basis arm along +e_2 with <x_1, w^*> = -31/60 so |eps_1| = 0.3500 at rho = 0.30
    c1_max_err = -31.0 / 60.0
    features[1, 0] = c1_max_err
    features[1, 1] = float(np.sqrt(1.0 - c1_max_err**2))

    # Arm 2: Companion arm along -e_2 with large basis coordinate norm so RLB Phase-1 tests e_2
    c1_comp = -0.30
    features[2, 0] = c1_comp
    features[2, 1] = -float(np.sqrt(1.0 - c1_comp**2))

    arm_ptr = 3
    # Basis arms for remaining coordinates j = 2, ..., d-1 (e_3, ..., e_d)
    c1_rem_basis = 0.35
    perp_rem_basis = float(np.sqrt(1.0 - c1_rem_basis**2))
    for j in range(2, d):
        if arm_ptr < k - 1:
            features[arm_ptr, 0] = c1_rem_basis
            features[arm_ptr, j] = perp_rem_basis
            arm_ptr += 1
        if arm_ptr < k - 1:
            features[arm_ptr, 0] = c1_rem_basis
            features[arm_ptr, j] = -perp_rem_basis
            arm_ptr += 1

    num_basis_arms = arm_ptr
    num_grid = k - num_basis_arms

    # Grid of suboptimal arms covering <x_i, w^*> in [-0.40, 0.74] in span(e_1, e_3, ..., e_d)
    c1_grid = np.linspace(0.74, -0.40, num_grid, dtype=np.float64)
    for offset, c1 in enumerate(c1_grid):
        arm_idx = num_basis_arms + offset
        rad = float(np.sqrt(max(0.0, 1.0 - float(c1) ** 2)))
        features[arm_idx, 0] = c1
        if d > 2:
            perp = rng.normal(0.0, 1.0, size=d - 2)
            perp /= float(np.linalg.norm(perp))
            features[arm_idx, 2:] = rad * perp
        else:
            features[arm_idx, 1] = rad

    linear_means = features @ w_star
    f_star = float(linear_means[0])  # 1.0
    linear_gaps = f_star - linear_means

    misspecification = np.zeros(k, dtype=np.float64)
    for i in range(1, k):
        if i in (1, 2):
            s_i = 1.0
        elif float(linear_gaps[i]) > 0.75:
            # Ensure |eps_i| <= 0.3500 so ||eps||_inf is attained exactly at Arm 1 (0.3500)
            s_i = 1.0
        elif skew_suboptimal_bias:
            coord_idx = 2 if d > 2 else 1
            s_i = 1.0 if features[i, coord_idx] >= 0.0 else -1.0
        else:
            s_i = 1.0 if (i % 2 == 1) else -1.0

        true_gap_i = float(linear_gaps[i]) / (1.0 + s_i * rho)
        true_mean_i = f_star - true_gap_i
        misspecification[i] = true_mean_i - float(linear_means[i])

    if permute_arms:
        perm = rng.permutation(k)
        features = features[perm]
        misspecification = misspecification[perm]

    return LinearBanditEnvironment(
        name=f"Regime 3: Liu rho-GAM (rho={rho:.2f}, k={k}, d={d})",
        features=features,
        theta_star=w_star,
        misspecification=misspecification,
        sigma=sigma,
        expected_rho_gam=rho,
    )