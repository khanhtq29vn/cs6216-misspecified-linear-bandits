"""
Numerical linear algebra and optimal experimental design utilities for linear bandits.

All mathematical operations strictly adhere to the notation conventions in
Prof. Jonathan Scarlett's CS6216 Mathematical Background notes:
- Column vectors x in R^d are represented as 1D NumPy arrays of shape (d,), or stacked
  row-wise in a feature matrix X in R^{k x d} where row i is x_i^T.
- Inner products are <u, v> = u^T v; Euclidean norms are ||u||_2;
  Mahalanobis norms are ||u||_A = sqrt(u^T A u) for symmetric positive semidefinite A.
- All logarithms are natural logarithms (base e).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class OptimalDesignResult:
    """
    Immutable output container for the Kiefer-Wolfowitz G-optimal design solver.

    Attributes
    ----------
    pi : FloatArray of shape (k_active,)
        Optimal probability distribution over the active arm set A_l summing to 1.
    max_variance : float
        Achieved worst-case directional variance g(pi) = max_{x in A_l} ||x||_{V(pi)^+}^2.
    effective_rank : int
        Dimension r = rank(span(A_l)) <= d of the active feature subspace.
    duality_gap : float
        Relative Kiefer-Wolfowitz duality gap (max_variance - r) / r >= 0.
    num_iterations : int
        Number of Frank-Wolfe (Fedorov-Wynn) iterations executed.
    """

    pi: FloatArray
    max_variance: float
    effective_rank: int
    duality_gap: float
    num_iterations: int


def sherman_morrison_update(V_inv: FloatArray, x: FloatArray) -> FloatArray:
    """
    Perform a rank-1 Sherman-Morrison inverse update in O(d^2) operations:
        (V + x x^T)^{-1} = V^{-1} - (V^{-1} x x^T V^{-1}) / (1 + x^T V^{-1} x).

    Enforces exact matrix symmetry `0.5 * (V_new + V_new^T)` at every step to
    eliminate floating-point asymmetry drift over long horizons (n >= 6,000).

    Parameters
    ----------
    V_inv : FloatArray of shape (d, d)
        Symmetric positive definite inverse design matrix V_{t-1}^{-1}.
    x : FloatArray of shape (d,)
        Pulled feature vector X_t.

    Returns
    -------
    FloatArray of shape (d, d)
        Updated symmetric positive definite inverse matrix V_t^{-1}.
    """
    if V_inv.ndim != 2 or V_inv.shape[0] != V_inv.shape[1]:
        raise ValueError(f"V_inv must be a square 2D array, got shape {V_inv.shape}.")
    if x.ndim != 1 or x.shape[0] != V_inv.shape[0]:
        raise ValueError(
            f"Dimension mismatch: V_inv has shape {V_inv.shape}, x has shape {x.shape}."
        )

    Vx: FloatArray = V_inv @ x
    quad_form: float = float(np.dot(x, Vx))
    if quad_form < -1e-12:
        raise ValueError(
            f"Non-positive definite matrix encountered in Sherman-Morrison: x^T V^-1 x = {quad_form}."
        )
    denom: float = 1.0 + max(0.0, quad_form)
    V_inv_new: FloatArray = V_inv - np.outer(Vx, Vx) / denom
    return 0.5 * (V_inv_new + V_inv_new.T)


def batch_mahalanobis_norm(X: FloatArray, A: FloatArray) -> FloatArray:
    """
    Compute the Mahalanobis norm ||x_i||_A = sqrt(x_i^T A x_i) for every row x_i of X.

    Parameters
    ----------
    X : FloatArray of shape (k, d)
        Matrix of k feature vectors in R^d.
    A : FloatArray of shape (d, d)
        Symmetric positive semidefinite matrix (e.g., V_t^{-1} or V_l^+).

    Returns
    -------
    FloatArray of shape (k,)
        Vector of norms (||x_0||_A, ..., ||x_{k-1}||_A)^T.
    """
    if X.ndim != 2 or A.ndim != 2 or X.shape[1] != A.shape[0]:
        raise ValueError(
            f"Incompatible shapes for batch_mahalanobis_norm: X={X.shape}, A={A.shape}."
        )
    quad_forms: FloatArray = np.einsum("ij,jk,ik->i", X, A, X, optimize=True)
    return np.sqrt(np.maximum(0.0, quad_forms))


def select_well_conditioned_basis(
    X: FloatArray, tol: float = 1e-8
) -> Tuple[IntArray, IntArray]:
    """
    Select a well-conditioned basis B of r = rank(X) <= d linearly independent arms
    from the feature matrix X in R^{k x d} using Column-Pivoted QR factorization
    on X^T in R^{d x k}.

    Mathematical Justification:
    In Phase 1 of the Robust Linear Bandit (RLB) algorithm (Ghosh et al., 2017, Sec. 4;
    Section 3.3 of `main.tex`), any test arm x_j outside B is expressed as x_j = X_B c_j
    where X_B = [x_{b_1} | ... | x_{b_r}] has the basis vectors as columns, and the
    hypothesis-test threshold w_{tau, j}(delta) scales with sqrt(||c_j||_2^2 + 1)
    where c_j = X_B^{-1} x_j. Selecting B via pivoted QR maximizes |det(X_B)| and
    minimizes cond(X_B), keeping ||c_j||_2 small and maximizing test power.

    Parameters
    ----------
    X : FloatArray of shape (k, d)
        Feature matrix whose rows are the k arm feature vectors x_i^T in R^d.
    tol : float
        Diagonal R threshold for numerical rank determination.

    Returns
    -------
    basis_indices : IntArray of shape (r,)
        Sorted indices of the r = rank(X) <= d selected basis arms.
    non_basis_indices : IntArray of shape (k - r,)
        Sorted indices of the remaining k - r arms outside the basis.
    """
    if X.ndim != 2:
        raise ValueError(f"Feature matrix X must be 2D, got shape {X.shape}.")
    k, d = X.shape

    _, R, piv = qr(X.T, mode="economic", pivoting=True)
    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return np.array([], dtype=np.int64), np.arange(k, dtype=np.int64)

    cutoff = tol * max(1.0, float(diag_R[0]))
    rank_eff = int(np.sum(diag_R > cutoff))
    rank_eff = min(rank_eff, d, k)

    basis_indices = np.sort(piv[:rank_eff].astype(np.int64))
    mask = np.ones(k, dtype=bool)
    mask[basis_indices] = False
    non_basis_indices = np.where(mask)[0].astype(np.int64)
    return basis_indices, non_basis_indices


def compute_g_optimal_design(
    arm_features: FloatArray,
    tol: float = 1e-5,
    max_iter: int = 1000,
    weight_prune_tol: float = 1e-5,
) -> OptimalDesignResult:
    """
    Compute an approximate G-optimal design distribution pi in Delta(A_l) over a finite
    active arm set A_l subset R^d via the Frank-Wolfe (Fedorov-Wynn) algorithm with
    pivoted-QR support initialization.

    Mathematical Guarantees (Kiefer-Wolfowitz Theorem, Theorem 2.5 in `main.tex`):
    Let V(pi) = sum_{x in A_l} pi(x) x x^T and r = rank(span(A_l)) <= d. Then
        min_{pi in Delta(A_l)} max_{x in A_l} x^T V(pi)^+ x = r <= d.
    At iteration m with current design pi^{(m)} and worst-case arm x_m^* achieving
    g_m = max_{x in A_l} x^T V(pi^{(m)})^+ x, maximizing log det_+ V((1-gamma)pi^{(m)} + gamma e_{x_m^*})
    yields the exact closed-form step size:
        gamma_m = (g_m / r - 1) / (g_m - 1) in (0, 1).

    Parameters
    ----------
    arm_features : FloatArray of shape (k_active, d)
        Feature matrix of currently active arms.
    tol : float
        Relative duality gap tolerance: terminates when g_m <= r * (1 + tol).
    max_iter : int
        Maximum number of Frank-Wolfe iterations.
    weight_prune_tol : float
        Threshold for pruning negligible support weights while preserving g(pi).

    Returns
    -------
    OptimalDesignResult
        Immutable dataclass containing pi, max_variance, effective_rank, duality_gap,
        and num_iterations.
    """
    if arm_features.ndim != 2 or arm_features.shape[0] == 0:
        raise ValueError(
            f"arm_features must be a non-empty 2D array, got shape {arm_features.shape}."
        )

    k_active, d = arm_features.shape
    if k_active == 1:
        return OptimalDesignResult(
            pi=np.array([1.0], dtype=np.float64),
            max_variance=1.0,
            effective_rank=1,
            duality_gap=0.0,
            num_iterations=0,
        )

    basis_idx, _ = select_well_conditioned_basis(arm_features, tol=1e-8)
    rank_eff = max(1, min(len(basis_idx), d))

    pi = np.zeros(k_active, dtype=np.float64)
    if len(basis_idx) > 0:
        pi[basis_idx] = 0.80 / float(len(basis_idx))
        pi += 0.20 / float(k_active)
    else:
        pi.fill(1.0 / float(k_active))
    pi /= np.sum(pi)

    max_var = float(rank_eff)
    num_iter = 0

    for it in range(1, max_iter + 1):
        num_iter = it
        V = (arm_features.T * pi) @ arm_features
        V_pinv = np.linalg.pinv(V, rcond=1e-10, hermitian=True)
        variances = batch_mahalanobis_norm(arm_features, V_pinv) ** 2

        j_max = int(np.argmax(variances))
        max_var = float(variances[j_max])

        if max_var <= float(rank_eff) * (1.0 + tol) or max_var <= 1.0 + 1e-12:
            break

        step = (max_var / float(rank_eff) - 1.0) / (max_var - 1.0)
        step = float(np.clip(step, 0.0, 1.0))

        pi = (1.0 - step) * pi
        pi[j_max] += step

    cand_pi = pi.copy()
    cand_pi[cand_pi < weight_prune_tol / float(k_active)] = 0.0
    if np.sum(cand_pi) > 0:
        cand_pi /= np.sum(cand_pi)
        V_cand = (arm_features.T * cand_pi) @ arm_features
        V_cand_pinv = np.linalg.pinv(V_cand, rcond=1e-10, hermitian=True)
        cand_max_var = float(
            np.max(batch_mahalanobis_norm(arm_features, V_cand_pinv) ** 2)
        )
        if cand_max_var <= float(rank_eff) * (1.0 + 5.0 * tol):
            pi = cand_pi
            max_var = cand_max_var

    duality_gap = max(0.0, (max_var - float(rank_eff)) / float(rank_eff))
    return OptimalDesignResult(
        pi=pi,
        max_variance=max_var,
        effective_rank=rank_eff,
        duality_gap=duality_gap,
        num_iterations=num_iter,
    )