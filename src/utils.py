"""
Mathematical utilities for stochastic and misspecified linear bandits.

Conventions follow Prof. Jonathan Scarlett's CS6216 mathematical notation:
- Feature vectors x in R^d are treated as column vectors in mathematical derivations
  and stored as 1D NumPy arrays of shape (d,) (or rows of X in R^{k x d}).
- Natural logarithms (base e) are used throughout.
"""

from typing import Tuple
import numpy as np


def sherman_morrison_update(V_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform a rank-1 Sherman-Morrison inverse update in O(d^2) time:
        (V + x x^T)^{-1} = V^{-1} - (V^{-1} x x^T V^{-1}) / (1 + x^T V^{-1} x)

    Parameters
    ----------
    V_inv : np.ndarray of shape (d, d)
        Current inverse of the regularized design matrix V_{t-1}^{-1}.
    x : np.ndarray of shape (d,)
        Selected action feature vector X_t.

    Returns
    -------
    np.ndarray of shape (d, d)
        Updated inverse matrix V_t^{-1}.
    """
    Vx = V_inv @ x
    denom = 1.0 + float(np.dot(x, Vx))
    return V_inv - np.outer(Vx, Vx) / denom


def batch_mahalanobis_norm(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute the Mahalanobis norm ||x_i||_A = sqrt(x_i^T A x_i) for each row x_i of X.

    Parameters
    ----------
    X : np.ndarray of shape (k, d)
        Matrix of arm feature vectors.
    A : np.ndarray of shape (d, d)
        Positive semi-definite matrix (e.g., V_t^{-1}).

    Returns
    -------
    np.ndarray of shape (k,)
        Vector of norms ||x_i||_A for i = 0, ..., k-1.
    """
    quad_forms = np.sum((X @ A) * X, axis=1)
    return np.sqrt(np.maximum(0.0, quad_forms))


def select_linearly_independent_basis(
    X: np.ndarray, d: int, tol: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a well-conditioned basis of d linearly independent arms from X in R^{k x d}
    along with remaining candidate arms outside the basis, used in Phase 1 of the
    Robust Linear Bandit (RLB) algorithm of Ghosh et al. (2017).

    Uses greedy Gram-Schmidt / maximum orthogonal projection selection to maximize
    the smallest singular value of the selected basis matrix X_B.

    Parameters
    ----------
    X : np.ndarray of shape (k, d)
        Feature matrix of all k arms.
    d : int
        Dimension of the ambient feature space.
    tol : float
        Numerical threshold for linear independence.

    Returns
    -------
    basis_indices : np.ndarray of shape (d_eff,)
        Indices of the selected basis arms (d_eff = rank(X) <= d).
    non_basis_indices : np.ndarray
        Indices of the remaining k - d_eff arms not in the basis.
    """
    k = X.shape[0]
    basis_list = []
    orthonormal_cols = []

    for _ in range(min(k, d)):
        best_idx = -1
        best_res_norm = -1.0
        best_vec = None

        for idx in range(k):
            if idx in basis_list:
                continue
            v = X[idx].astype(float).copy()
            for q in orthonormal_cols:
                v = v - np.dot(q, v) * q
            res_norm = float(np.linalg.norm(v))
            if res_norm > best_res_norm:
                best_res_norm = res_norm
                best_idx = idx
                best_vec = v

        if best_idx == -1 or best_res_norm <= tol:
            break

        basis_list.append(best_idx)
        orthonormal_cols.append(best_vec / best_res_norm)

    basis_indices = np.array(basis_list, dtype=int)
    mask = np.ones(k, dtype=bool)
    mask[basis_indices] = False
    non_basis_indices = np.where(mask)[0]
    return basis_indices, non_basis_indices


def compute_g_optimal_design(
    arm_features: np.ndarray, tol: float = 1e-4, max_iter: int = 500
) -> Tuple[np.ndarray, float]:
    """
    Compute an approximate G-optimal experimental design distribution pi over a finite
    active arm set A_l subset R^d using the Frank-Wolfe (Fedorov-Wynn) algorithm.

    By the Kiefer-Wolfowitz Equivalence Theorem (Lattimore & Szepesvari, Theorem 21.1),
    minimizing g(pi) = max_{x in A_l} x^T V(pi)^+ x over the probability simplex
    Delta(A_l), where V(pi) = sum_{x in A_l} pi(x) x x^T, achieves
    min_pi g(pi) = rank(span(A_l)) <= d.

    Parameters
    ----------
    arm_features : np.ndarray of shape (k_active, d)
        Feature vectors of currently active arms.
    tol : float
        Relative optimality tolerance (stops when max_var <= r * (1 + tol)).
    max_iter : int
        Maximum number of Frank-Wolfe iterations.

    Returns
    -------
    pi : np.ndarray of shape (k_active,)
        Probability distribution over the active arms summing to 1.
    max_var : float
        Achieved maximum prediction variance g(pi).
    """
    k_active, d = arm_features.shape
    if k_active == 1:
        return np.array([1.0], dtype=float), 1.0

    # Determine effective rank of the active feature subspace
    rank_eff = int(np.linalg.matrix_rank(arm_features, tol=1e-7))
    rank_eff = max(1, min(rank_eff, d))

    pi = np.ones(k_active, dtype=float) / float(k_active)
    max_var = float(rank_eff)

    for _ in range(max_iter):
        V = arm_features.T @ (arm_features * pi[:, None])
        V_pinv = np.linalg.pinv(V, rcond=1e-9)
        variances = np.sum((arm_features @ V_pinv) * arm_features, axis=1)
        j_max = int(np.argmax(variances))
        max_var = float(variances[j_max])

        if max_var <= rank_eff * (1.0 + tol) or max_var <= 1.0 + 1e-9:
            break

        # Exact line-search step size for the log-determinant D-optimal objective
        step = (max_var / float(rank_eff) - 1.0) / (max_var - 1.0)
        step = float(np.clip(step, 0.0, 1.0))

        pi = (1.0 - step) * pi
        pi[j_max] += step

    # Clean tiny weights and renormalize
    pi = np.maximum(0.0, pi)
    pi[pi < 1e-4 / k_active] = 0.0
    pi /= np.sum(pi)
    return pi, max_var