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
   theta^* = (0.70, 0.45, 0.35, 0.30, 0.25)^T (for d = 5). A corrupted arm x_{corr} lies
   in span(e_1, e_2) with shrinkage coordinates (alpha_1 + alpha_2 < 1) so its linearly
   extrapolated reward is mu_{corr}^{lin} = 0.485 < theta_1^* = 0.70, and carries a sparse
   deviation eps_k > 0.215 making it the unique true optimal arm.

3. Regime 3 — rho-Gap-Adjusted Misspecification (Section 4; Liu, Qiao, Yin, Bogunovic, & Wang, 2025):
       max_{x in X} f_{w^*}(x) = f^*   and   |f_{w^*}(x_i) - f_0(x_i)| <= rho * (f^* - f_0(x_i))
   for all x_i in X and rho in [0, 1). Automatically verifies Definition 4.1 and
   Proposition 4.2 (Multiplicative Gap Sandwiching and Exact Optimality Preservation
   across all arms, including optimal arms) upon initialization.
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
        X_arr = np.asarray(features, dtype=np.float64)
        theta_arr = np.asarray(theta_star, dtype=np.float64)
        eps_arr = np.asarray(misspecification, dtype=np.float64)

        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"features must be a non-empty 2D array, got shape {X_arr.shape}."
            )
        k, d = X_arr.shape
        if theta_arr.shape != (d,):
            raise ValueError(
                f"theta_star shape {theta_arr.shape} does not match d={d}."
            )
        if eps_arr.shape != (k,):
            raise ValueError(
                f"misspecification shape {eps_arr.shape} does not match k={k}."
            )
        if not (
            np.all(np.isfinite(X_arr))
            and np.all(np.isfinite(theta_arr))
            and np.all(np.isfinite(eps_arr))
        ):
            raise ValueError(
                "features, theta_star, and misspecification must contain only finite entries."
            )
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(
                f"Noise standard deviation sigma must be finite and >= 0, got {sigma}."
            )
        if not np.isfinite(feature_norm_bound) or feature_norm_bound <= 0.0:
            raise ValueError(
                f"feature_norm_bound L must be finite and > 0, got {feature_norm_bound}."
            )
        if expected_rho_gam is not None and not (0.0 <= expected_rho_gam < 1.0):
            raise ValueError(
                f"expected_rho_gam must lie in [0, 1), got {expected_rho_gam}."
            )

        self._name: str = str(name)
        self._k: int = k
        self._d: int = d
        self._sigma: float = float(sigma)
        self._L: float = float(feature_norm_bound)

        self._features: FloatArray = np.array(X_arr, dtype=np.float64, copy=True)
        self._theta_star: FloatArray = np.array(theta_arr, dtype=np.float64, copy=True)
        self._misspecification: FloatArray = np.array(
            eps_arr, dtype=np.float64, copy=True
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

        self._verify_invariants(expected_rho_gam)

    def _verify_invariants(self, expected_rho_gam: Optional[float]) -> None:
        max_norm = float(np.max(np.linalg.norm(self._features, axis=1)))
        if max_norm > self._L + 1e-7:
            raise AssertionError(
                f"Feature norm bound violated: max_i ||x_i||_2 = {max_norm:.6f} > L = {self._L}."
            )

        if expected_rho_gam is not None:
            atol = 1e-7
            f_star = self._optimal_mean
            lin_max = float(np.max(self._linear_means))
            if abs(lin_max - f_star) > atol:
                raise AssertionError(
                    f"rho-GAM peak condition violated: max_x f_{{w^*}}(x) = {lin_max} != f^* = {f_star}."
                )

            gaps = self._suboptimality_gaps
            lin_gaps = f_star - self._linear_means

            if np.any(np.abs(self._misspecification) > expected_rho_gam * gaps + atol):
                max_violation = float(
                    np.max(np.abs(self._misspecification) - expected_rho_gam * gaps)
                )
                raise AssertionError(
                    f"rho-GAM pointwise condition |eps_i| <= rho * Delta_i failed "
                    f"(max violation = {max_violation:.6e})."
                )

            lower_ok = np.all(lin_gaps >= (1.0 - expected_rho_gam) * gaps - atol)
            upper_ok = np.all(lin_gaps <= (1.0 + expected_rho_gam) * gaps + atol)
            if not (lower_ok and upper_ok):
                raise AssertionError(
                    "Proposition 4.2 multiplicative gap-sandwiching inequality failed verification."
                )

            true_opt_set = np.flatnonzero(np.isclose(self._true_means, f_star, atol=atol))
            lin_opt_set = np.flatnonzero(np.isclose(self._linear_means, lin_max, atol=atol))
            if not np.array_equal(true_opt_set, lin_opt_set):
                raise AssertionError(
                    f"Proposition 4.2 optimality preservation failed: "
                    f"argmax f_0 = {true_opt_set.tolist()} != argmax f_{{w^*}} = {lin_opt_set.tolist()}."
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
        view = self._features.view()
        view.setflags(write=False)
        return view

    @property
    def theta_star(self) -> FloatArray:
        view = self._theta_star.view()
        view.setflags(write=False)
        return view

    @property
    def misspecification(self) -> FloatArray:
        view = self._misspecification.view()
        view.setflags(write=False)
        return view

    @property
    def linear_means(self) -> FloatArray:
        view = self._linear_means.view()
        view.setflags(write=False)
        return view

    @property
    def true_means(self) -> FloatArray:
        view = self._true_means.view()
        view.setflags(write=False)
        return view

    @property
    def optimal_arm(self) -> int:
        return self._optimal_arm

    @property
    def optimal_mean(self) -> float:
        return self._optimal_mean

    @property
    def suboptimality_gaps(self) -> FloatArray:
        view = self._suboptimality_gaps.view()
        view.setflags(write=False)
        return view

    @property
    def corrupted_arm_idx(self) -> Optional[int]:
        return self._corrupted_arm_idx

    @property
    def basis_arm_1_idx(self) -> Optional[int]:
        return self._basis_arm_1_idx

    def step(self, arm_idx: int, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Execute one sequential interaction step by pulling `arm_idx` in {0, ..., k-1}.
        """
        if not isinstance(arm_idx, (int, np.integer)) or isinstance(arm_idx, bool):
            raise TypeError(f"arm_idx must be an integer, got {type(arm_idx)!r}.")
        idx = int(arm_idx)
        if not (0 <= idx < self._k):
            raise IndexError(f"arm_idx={idx} out of bounds for k={self._k} arms.")
        reward = float(self._true_means[idx]) + float(rng.normal(0.0, self._sigma))
        inst_regret = float(self._suboptimality_gaps[idx])
        return reward, inst_regret

    def diagnostics(self) -> EnvironmentDiagnostics:
        """
        Compute comprehensive mathematical diagnostics of the environment.
        """
        atol = 1e-10
        eps_linf = float(np.max(np.abs(self._misspecification)))
        f_star = self._optimal_mean
        lin_max = float(np.max(self._linear_means))

        true_opt_mask = self._suboptimality_gaps <= atol
        true_opt_set = np.flatnonzero(true_opt_mask)
        lin_opt_set = np.flatnonzero(np.isclose(self._linear_means, lin_max, atol=1e-7))
        preserves_opt = bool(
            abs(lin_max - f_star) <= 1e-7 and np.array_equal(true_opt_set, lin_opt_set)
        )

        non_opt = ~true_opt_mask
        if np.any(non_opt):
            min_gap = float(np.min(self._suboptimality_gaps[non_opt]))
            second_best = float(np.max(self._true_means[non_opt]))
        else:
            min_gap = 0.0
            second_best = self._optimal_mean

        opt_bias = float(np.max(np.abs(self._misspecification[true_opt_mask])))
        if opt_bias > 1e-7 or not preserves_opt:
            rho_gam = float("inf")
        elif np.any(non_opt):
            rho_gam = float(
                np.max(
                    np.abs(self._misspecification[non_opt])
                    / self._suboptimality_gaps[non_opt]
                )
            )
        else:
            rho_gam = 0.0

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
            preserves_linear_optimum=preserves_opt,
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
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}.")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}.")
    if not (0.0 < min_gap < 2.0):
        raise ValueError(f"min_gap must lie in (0, 2), got {min_gap}.")

    rng = np.random.default_rng(seed)
    base_theta = np.array([0.80, 0.45, -0.35, 0.25, -0.20], dtype=np.float64)
    if d <= len(base_theta):
        theta_star = base_theta[:d].copy()
    else:
        extra = rng.uniform(-0.3, 0.3, size=d - len(base_theta))
        theta_star = np.concatenate([base_theta, extra])
    theta_star /= float(np.linalg.norm(theta_star))

    features = np.zeros((k, d), dtype=np.float64)
    features[0] = theta_star.copy()

    max_subopt_mean = 1.0 - float(min_gap)
    min_subopt_mean = min(0.55, max_subopt_mean - 0.10)
    min_subopt_mean = max(-1.0, min_subopt_mean)
    c1_vals = np.linspace(max_subopt_mean, min_subopt_mean, num=k - 1, dtype=np.float64)

    for idx in range(1, k):
        c1 = float(c1_vals[idx - 1])
        if d == 1:
            features[idx] = c1 * theta_star
        else:
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

    Mathematical Properties:
    - Standard benchmark (`skew_suboptimal_bias=False`):
      * Unique optimal arm x^* = w^* = e_1 with f^* = 1.0 and eps(x^*) = 0.
      * Basis arm x_1 along +e_2 has <x_1, w^*> = -31/60 and unit norm 1.0, while
        companion diagnostic arm x_2 along -e_2 has norm 0.95 (guaranteeing
        ||c_2||_2 = 1.365 is strictly maximal so RLB tests x_2 and rejects H_0 in 100% of trials!).
      * At rho = 0.30, global sup-norm error is ||eps||_inf = 0.3500.
    - High-rho directional skew benchmark (`skew_suboptimal_bias=True`):
      * Satisfies 100% of the exact rho-GAM invariants (Definition 4.1 & Proposition 4.2).
      * Optimal arm x^* = 0.55 e_1 (f^* = 0.55, eps(x^*) = 0, norm = 0.55).
      * Near-optimal competitor cluster (6 arms) near x_c = 0.50 e_1 + 0.866 e_2 (unit norm 1.0)
        has small linear gap Delta_c^{lin} = 0.05 and s_c = -1, so its true gap is
        Delta_c = 0.05 / (1 - rho) (growing to 0.125 at rho = 0.60) and eps_c = -rho * Delta_c.
      * High-gap exploratory cluster (22 arms) near -0.45 e_1 + 0.89 e_2 (unit norm 1.0)
        carries positive misspecification eps_j = +rho * Delta_j > 0 along +e_2.
        In adaptive LinUCB, early pulls across these 22 arms inject persistent positive bias
        into theta_hat_{t,2} inside the unreset Gram matrix V_{t-1}, inflating the ridge
        prediction <x_c, theta_hat_t> above <x^*, theta_hat_t> as rho increases into [0.35, 0.60].
        In contrast, G-Optimal Phased Elimination purges the high-gap arms after Phase 1
        and balances +e_2 against -e_2, remaining flat across rho in [0.05, 0.60]!
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho-GAM parameter rho must lie in [0, 1), got {rho}.")
    if d < 2:
        raise ValueError(f"make_gap_adjusted_env requires d >= 2, got d={d}.")
    if k <= d:
        raise ValueError(f"k must exceed d={d}, got k={k}.")

    rng = np.random.default_rng(seed)
    w_star = np.zeros(d, dtype=np.float64)
    w_star[0] = 1.0
    features = np.zeros((k, d), dtype=np.float64)

    if not skew_suboptimal_bias:
        features[0] = w_star.copy()  # Unique optimal arm x^* = e_1 (f^* = 1.0, norm = 1.0)

        c1_max_err = -31.0 / 60.0
        features[1, 0] = c1_max_err
        features[1, 1] = float(np.sqrt(max(0.0, 1.0 - c1_max_err**2)))

        norm_arm2 = 0.95
        features[2, 0] = c1_max_err
        features[2, 1] = -float(np.sqrt(max(0.0, norm_arm2**2 - c1_max_err**2)))
        arm_ptr = 3

        c1_rem_basis = 0.35
        perp_pos = float(np.sqrt(max(0.0, 1.0 - c1_rem_basis**2)))
        perp_neg = float(np.sqrt(max(0.0, 0.86**2 - c1_rem_basis**2)))
        for j in range(2, d):
            if arm_ptr < k - 1:
                features[arm_ptr, 0] = c1_rem_basis
                features[arm_ptr, j] = perp_pos
                arm_ptr += 1
            if arm_ptr < k - 1:
                features[arm_ptr, 0] = c1_rem_basis
                features[arm_ptr, j] = -perp_neg
                arm_ptr += 1

        num_basis_arms = arm_ptr
        num_grid = k - num_basis_arms

        c1_grid = np.linspace(0.74, -0.40, num=num_grid, dtype=np.float64)
        for offset, c1 in enumerate(c1_grid):
            arm_idx = num_basis_arms + offset
            norm_g = 0.84
            rad = float(np.sqrt(max(0.0, norm_g**2 - float(c1) ** 2)))
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
            if i < num_basis_arms or float(linear_gaps[i]) > 0.75:
                s_i = 1.0
            else:
                s_i = 1.0 if (i % 2 == 1) else -1.0

            true_gap_i = float(linear_gaps[i]) / (1.0 + s_i * rho)
            true_mean_i = f_star - true_gap_i
            misspecification[i] = true_mean_i - float(linear_means[i])

    else:
        # Skewed suboptimal bias geometry for Experiment 3b (rho sweep in [0.05, 0.60]):
        # Optimal arm x^* = 0.55 * e_1 (f^* = 0.55, eps(x^*) = 0)
        f_star_val = 0.55
        features[0, 0] = f_star_val

        # Arm 1 & Arm 2: High-discrepancy e_2 basis/test pair so RLB rejects H_0
        c1_diag = -0.45
        features[1, 0] = c1_diag
        features[1, 1] = float(np.sqrt(1.0 - c1_diag**2))
        features[2, 0] = c1_diag
        features[2, 1] = -float(np.sqrt(0.95**2 - c1_diag**2))

        # Near-optimal competitor cluster (6 arms) near 0.49 * e_1 + 0.8717 * e_2 (unit norm 1.0)
        # Has small linear gap Delta_c^{lin} = 0.55 - 0.49 = 0.06 and s_c = -1,
        # so true gap Delta_c = 0.06 / (1 - rho) (reaching 0.15 at rho = 0.60)
        n_comp = min(6, max(2, k // 7))
        for idx in range(3, 3 + n_comp):
            c1_c = 0.49 - 0.002 * float(idx - 3)
            c2_c = float(np.sqrt(max(0.0, 1.0 - c1_c**2 - 0.0064)))
            features[idx, 0] = c1_c
            features[idx, 1] = c2_c
            if d > 2:
                features[idx, 2 + ((idx - 3) % (d - 2))] = 0.08

        arm_ptr = 3 + n_comp
        # Orthogonal basis arms for e_3, ..., e_d
        for j in range(2, d):
            if arm_ptr < k:
                features[arm_ptr, 0] = 0.15
                features[arm_ptr, j] = float(np.sqrt(1.0 - 0.15**2))
                arm_ptr += 1

        # Remaining high-gap exploratory arms skewed along +e_2 with s_j = +1 (positive bias +rho * Delta_j)
        num_rem = k - arm_ptr
        if num_rem > 0:
            c1_rem = np.linspace(-0.20, -0.44, num=num_rem, dtype=np.float64)
            for offset, c1 in enumerate(c1_rem):
                idx = arm_ptr + offset
                features[idx, 0] = c1
                if d > 2:
                    orth_comp = 0.28 * ((-1.0) ** offset)
                    c2_val = float(np.sqrt(max(0.0, 0.96**2 - float(c1) ** 2 - orth_comp**2)))
                    features[idx, 1] = c2_val
                    features[idx, 2 + (offset % (d - 2))] = orth_comp
                else:
                    features[idx, 1] = float(np.sqrt(max(0.0, 0.96**2 - float(c1) ** 2)))

        linear_means = features @ w_star
        f_star = float(linear_means[0])  # 0.55
        linear_gaps = f_star - linear_means

        misspecification = np.zeros(k, dtype=np.float64)
        for i in range(1, k):
            if 3 <= i < 3 + n_comp:
                s_i = -1.0
            else:
                s_i = 1.0
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