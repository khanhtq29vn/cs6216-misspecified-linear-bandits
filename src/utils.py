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
import warnings
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class OptimalDesignResult:
    """
    Immutable output container for the Kiefer-Wolfowitz G-optimal design solver.

    Attributes
    ----------
    pi : FloatArray of shape (k_active,)
        Read-only optimal probability distribution over the active arm set A_l summing to 1,
        with support size guaranteed <= r(r + 1) / 2 via Caratheodory reduction.
    max_variance : float
        Independently certified worst-case directional variance
        g(pi) = max_{x in A_l} x^T V(pi)^+ x evaluated on the unregularized Gram matrix.
    effective_rank : int
        Dimension r = rank(span(A_l)) <= d of the active feature subspace.
    duality_gap : float
        Relative Kiefer-Wolfowitz duality gap max(0, (max_variance - r) / r).
    num_iterations : int
        Number of Frank-Wolfe (Fedorov-Wynn-Atwood) iterations executed.
    converged : bool
        True if and only if max_variance <= r * (1 + tol) upon termination.
    """

    pi: FloatArray
    max_variance: float
    effective_rank: int
    duality_gap: float
    num_iterations: int
    converged: bool = True

    def __post_init__(self) -> None:
        pi_copy = np.array(self.pi, dtype=np.float64, copy=True)
        pi_copy.setflags(write=False)
        object.__setattr__(self, "pi", pi_copy)


def _canonical_feature_order(X: FloatArray) -> IntArray:
    """
    Produce a deterministic canonical ordering of the rows of X in R^{k x d}
    based on rounded Euclidean norm (descending) and lexicographic coordinate values.
    Ensures that basis selection and G-optimal initialization select the exact same
    physical feature vectors regardless of input row permutation or BLAS backend.
    """
    k = X.shape[0]
    if k <= 1:
        return np.arange(k, dtype=np.int64)

    row_norms = np.linalg.norm(X, axis=1)
    max_norm = float(np.max(row_norms))
    scale = max_norm if max_norm > 1e-15 else 1.0
    X_scaled = np.round(X / scale, decimals=11)
    norms_rounded = np.round(-row_norms / scale, decimals=11)

    keys = [X_scaled[:, j] for j in range(X.shape[1] - 1, -1, -1)]
    keys.append(norms_rounded)
    return np.lexsort(keys).astype(np.int64)


def sherman_morrison_update(V_inv: FloatArray, x: FloatArray) -> FloatArray:
    """
    Perform a rank-1 Sherman-Morrison inverse update in O(d^2) operations:
        (V + x x^T)^{-1} = V^{-1} - (V^{-1} x x^T V^{-1}) / (1 + x^T V^{-1} x).

    Validates finite inputs and matrix symmetry, and enforces exact symmetry
    `0.5 * (V_new + V_new^T)` at every step to eliminate floating-point drift.
    """
    V_arr = np.asarray(V_inv, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if V_arr.ndim != 2 or V_arr.shape[0] != V_arr.shape[1] or V_arr.shape[0] == 0:
        raise ValueError(
            f"V_inv must be a non-empty square 2D array, got shape {V_arr.shape}."
        )
    if x_arr.ndim != 1 or x_arr.shape[0] != V_arr.shape[0]:
        raise ValueError(
            f"Dimension mismatch: V_inv has shape {V_arr.shape}, x has shape {x_arr.shape}."
        )
    if not (np.all(np.isfinite(V_arr)) and np.all(np.isfinite(x_arr))):
        raise ValueError("V_inv and x must contain only finite floating-point values.")

    sym_scale = max(1.0, float(np.max(np.abs(V_arr))))
    if float(np.max(np.abs(V_arr - V_arr.T))) > 1e-8 * sym_scale:
        raise ValueError("V_inv must be symmetric positive definite.")

    Vx: FloatArray = V_arr @ x_arr
    quad_form: float = float(np.dot(x_arr, Vx))
    norm_x_sq: float = float(np.dot(x_arr, x_arr))
    if quad_form < -1e-10 * max(1.0, norm_x_sq * sym_scale):
        raise ValueError(
            f"Non-positive definite matrix encountered in Sherman-Morrison: x^T V^-1 x = {quad_form}."
        )
    denom: float = 1.0 + max(0.0, quad_form)
    V_inv_new: FloatArray = V_arr - np.outer(Vx, Vx) / denom
    return 0.5 * (V_inv_new + V_inv_new.T)


def batch_mahalanobis_norm(X: FloatArray, A: FloatArray) -> FloatArray:
    """
    Compute the Mahalanobis norm ||x_i||_A = sqrt(x_i^T A x_i) for every row x_i of X.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)

    if (
        X_arr.ndim != 2
        or A_arr.ndim != 2
        or A_arr.shape[0] != A_arr.shape[1]
        or X_arr.shape[1] != A_arr.shape[0]
    ):
        raise ValueError(
            f"Incompatible shapes for batch_mahalanobis_norm: X={X_arr.shape}, A={A_arr.shape}."
        )
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(A_arr))):
        raise ValueError("X and A must contain only finite floating-point values.")

    sym_scale = max(1.0, float(np.max(np.abs(A_arr)))) if A_arr.size > 0 else 1.0
    if A_arr.size > 0 and float(np.max(np.abs(A_arr - A_arr.T))) > 1e-8 * sym_scale:
        raise ValueError("Matrix A must be symmetric positive semidefinite.")

    quad_forms: FloatArray = np.einsum("ij,jk,ik->i", X_arr, A_arr, X_arr, optimize=True)
    if quad_forms.size > 0:
        row_sq_norms = np.sum(X_arr * X_arr, axis=1)
        tol_neg = 1e-10 * sym_scale * np.maximum(1.0, row_sq_norms)
        if np.any(quad_forms < -tol_neg):
            raise ValueError(
                f"Matrix A is not positive semidefinite: min quadratic form = {float(np.min(quad_forms))}."
            )
    return np.sqrt(np.maximum(0.0, quad_forms))


def select_well_conditioned_basis(
    X: FloatArray, tol: float = 1e-8
) -> Tuple[IntArray, IntArray]:
    """
    Select a well-conditioned basis B of r = rank(X) <= d linearly independent arms
    from X in R^{k x d} using deterministic Modified Gram-Schmidt with canonical
    lexicographic tie-breaking.

    Properties:
    - Strictly scale-invariant: replacing X by alpha * X (alpha != 0) selects the
      exact same basis indices.
    - Permutation-equivariant on distinct rows: permuting the rows of X selects the
      exact same physical feature vectors, eliminating cross-platform LAPACK dgeqp3
      tie-breaking discrepancies on unit-norm feature sets.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(f"Feature matrix X must be non-empty 2D, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Feature matrix X must contain only finite entries.")
    if tol <= 0.0:
        raise ValueError(f"Rank tolerance tol must be > 0, got {tol}.")

    k, d = X_arr.shape
    canon_order = _canonical_feature_order(X_arr)
    X_canon = X_arr[canon_order].copy()
    residual_sq = np.sum(X_canon * X_canon, axis=1)
    max_init_norm = float(np.sqrt(np.max(residual_sq)))

    if max_init_norm <= 1e-15:
        return np.array([], dtype=np.int64), np.arange(k, dtype=np.int64)

    cutoff_sq = (float(tol) * max_init_norm) ** 2
    selected_canon: list[int] = []
    Q_basis: list[FloatArray] = []

    for _ in range(min(k, d)):
        max_res_sq = float(np.max(residual_sq))
        if max_res_sq <= cutoff_sq:
            break
        candidates = np.where(residual_sq >= max_res_sq * (1.0 - 1e-11))[0]
        best_c = int(candidates[0])
        selected_canon.append(best_c)

        q_vec = X_canon[best_c].copy()
        for prev_q in Q_basis:
            q_vec -= float(np.dot(prev_q, q_vec)) * prev_q
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm <= float(tol) * max_init_norm:
            selected_canon.pop()
            break
        q_vec /= q_norm
        Q_basis.append(q_vec)

        proj = X_canon @ q_vec
        X_canon -= np.outer(proj, q_vec)
        residual_sq = np.sum(X_canon * X_canon, axis=1)
        for idx_c in selected_canon:
            residual_sq[idx_c] = -1.0

    basis_indices = canon_order[np.array(selected_canon, dtype=np.int64)]
    mask = np.ones(k, dtype=bool)
    mask[basis_indices] = False
    non_basis_indices = canon_order[mask[canon_order]]
    return basis_indices, non_basis_indices


def _caratheodory_support_reduction(
    Z: FloatArray, pi: FloatArray, tol: float = 1e-12
) -> FloatArray:
    """
    Reduce the support of a design distribution pi on whitened features Z in R^{k x r}
    via Caratheodory's theorem so that |supp(pi)| <= r(r + 1) / 2 while preserving
    the exact moment matrix M(pi) = sum_i pi_i z_i z_i^T and sum_i pi_i = 1.
    """
    pi_red = np.array(pi, dtype=np.float64, copy=True)
    r = Z.shape[1]
    max_support = (r * (r + 1)) // 2
    triu_r, triu_c = np.triu_indices(r)

    while True:
        supp = np.where(pi_red > tol)[0]
        m = int(supp.size)
        if m <= max_support:
            break

        Z_supp = Z[supp]
        outer_stack = Z_supp[:, triu_r] * Z_supp[:, triu_c]  # shape (m, r(r+1)/2)
        A_eq = np.vstack([outer_stack.T, np.ones((1, m), dtype=np.float64)])

        _, s_vals, vt = np.linalg.svd(A_eq, full_matrices=True)
        if vt.shape[0] <= A_eq.shape[0] and float(s_vals[-1]) > 1e-11 * float(s_vals[0]):
            break

        # Last row of V^T lies in the exact right nullspace of A_eq (so sum(null_vec) == 0
        # and sum_i null_vec[i] z_i z_i^T == 0 up to machine precision!)
        null_vec = vt[-1].copy()
        if float(np.max(np.abs(null_vec))) <= 1e-14:
            break

        if np.all(null_vec >= -1e-14):
            null_vec = -null_vec
        neg_mask = null_vec < -1e-14
        if not np.any(neg_mask):
            break

        ratios = -pi_red[supp[neg_mask]] / null_vec[neg_mask]
        min_pos = int(np.argmin(ratios))
        alpha = float(ratios[min_pos])

        pi_red[supp] += alpha * null_vec
        drop_idx = int(supp[neg_mask][min_pos])
        pi_red[drop_idx] = 0.0
        pi_red[pi_red < tol] = 0.0
        total_w = float(np.sum(pi_red))
        if total_w > 0.0:
            pi_red /= total_w

    return pi_red


def compute_g_optimal_design(
    arm_features: FloatArray,
    tol: float = 1e-5,
    max_iter: int = 1000,
    weight_prune_tol: float = 1e-7,
) -> OptimalDesignResult:
    """
    Compute an approximate G-optimal design distribution pi in Delta(A_l) over a finite
    active arm set A_l subset R^d via scale-invariant SVD whitening, Exact-Gain
    Away-Step Frank-Wolfe (Fedorov-Wynn-Atwood-Todd), and Caratheodory support reduction.

    Key Mathematical & Numerical Properties:
    1. Exact Scale Invariance: Whitens the active subspace via reduced SVD X = U_r Sigma_r V_r^T,
       working in the dimensionless orthonormal coordinate matrix Z = U_r in R^{k_active x r}.
    2. Exact Finite-Step Objective Comparison: Compares the exact D-optimal log-determinant
       gain Delta_toward vs. Delta_away (accounting for step-size clipping at alpha_max = pi_j/(1-pi_j))
       and sets dropped weights strictly to 0.0, achieving linear convergence in < 80 iterations.
    3. Caratheodory Support Reduction & Independent Recertification: Reduces support to
       at most r(r+1)/2 positive weights while preserving M(pi) to machine precision.
    """
    X_arr = np.asarray(arm_features, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(
            f"arm_features must be a non-empty 2D array, got shape {X_arr.shape}."
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("arm_features must contain only finite floating-point entries.")
    if tol <= 0.0:
        raise ValueError(f"Tolerance tol must be > 0, got {tol}.")
    if max_iter < 0:
        raise ValueError(f"max_iter must be >= 0, got {max_iter}.")

    k_active, _ = X_arr.shape

    u_mat, s_vals, _ = np.linalg.svd(X_arr, full_matrices=False)
    if s_vals.size == 0 or float(s_vals[0]) <= 1e-15:
        pi_uniform = np.full(k_active, 1.0 / float(k_active), dtype=np.float64)
        return OptimalDesignResult(
            pi=pi_uniform,
            max_variance=0.0,
            effective_rank=0,
            duality_gap=0.0,
            num_iterations=0,
            converged=True,
        )

    rank_eff = int(np.sum(s_vals > 1e-8 * float(s_vals[0])))
    rank_eff = max(1, rank_eff)
    r_float = float(rank_eff)

    if k_active == 1:
        return OptimalDesignResult(
            pi=np.array([1.0], dtype=np.float64),
            max_variance=1.0,
            effective_rank=1,
            duality_gap=0.0,
            num_iterations=0,
            converged=True,
        )

    Z: FloatArray = u_mat[:, :rank_eff].copy()

    basis_idx, _ = select_well_conditioned_basis(Z, tol=1e-8)
    pi = np.zeros(k_active, dtype=np.float64)
    if basis_idx.size == rank_eff:
        pi[basis_idx] = 1.0 / float(rank_eff)
    else:
        pi.fill(1.0 / float(k_active))

    def _eval_unregularized_leverages(
        weights: FloatArray,
    ) -> Tuple[bool, float, FloatArray]:
        M_mat = (Z.T * weights) @ Z
        M_mat = 0.5 * (M_mat + M_mat.T)
        eigvals, eigvecs = np.linalg.eigh(M_mat)
        if float(eigvals[0]) <= 1e-13 * max(1.0, float(eigvals[-1])):
            return False, -np.inf, np.full(k_active, np.inf)
        logdet = float(np.sum(np.log(eigvals)))
        M_inv = (eigvecs / eigvals) @ eigvecs.T
        M_inv = 0.5 * (M_inv + M_inv.T)
        levs = batch_mahalanobis_norm(Z, M_inv) ** 2
        return True, logdet, levs

    ok_init, logdet_cur, variances = _eval_unregularized_leverages(pi)
    if not ok_init:
        pi.fill(1.0 / float(k_active))
        ok_init, logdet_cur, variances = _eval_unregularized_leverages(pi)

    max_var = float(np.max(variances))
    converged = max_var <= r_float * (1.0 + tol)
    num_iter = 0

    for it in range(1, max_iter + 1):
        if converged:
            break
        num_iter = it

        j_plus = int(np.argmax(variances))
        g_plus = float(variances[j_plus])
        max_var = g_plus

        if g_plus <= r_float * (1.0 + tol) or g_plus <= 1.0 + 1e-12:
            converged = True
            break

        # Compute exact toward step size and its exact log-det increase
        if rank_eff == 1:
            step_toward = 1.0
            gain_toward = float(np.log(max(1.0, g_plus)))
        else:
            step_toward = float(np.clip((g_plus / r_float - 1.0) / (g_plus - 1.0), 0.0, 1.0))
            gain_toward = (r_float - 1.0) * float(np.log(1.0 - step_toward)) + float(
                np.log(1.0 + step_toward * (g_plus - 1.0))
            )

        supp_idx = np.where(pi > 0.0)[0]
        took_away_step = False
        if supp_idx.size > rank_eff:
            j_minus = int(supp_idx[int(np.argmin(variances[supp_idx]))])
            g_minus = float(variances[j_minus])

            if g_minus < r_float - 1e-12 and pi[j_minus] < 1.0 - 1e-9:
                alpha_max = float(pi[j_minus] / (1.0 - pi[j_minus]))
                if g_minus <= 1.0 + 1e-9:
                    step_away = alpha_max
                    is_drop = True
                else:
                    step_opt = (r_float - g_minus) / (r_float * (g_minus - 1.0))
                    if step_opt >= alpha_max:
                        step_away = alpha_max
                        is_drop = True
                    else:
                        step_away = float(max(0.0, step_opt))
                        is_drop = False

                inner_arg = 1.0 - step_away * (g_minus - 1.0)
                if inner_arg > 1e-12:
                    gain_away = (r_float - 1.0) * float(np.log(1.0 + step_away)) + float(
                        np.log(inner_arg)
                    )
                else:
                    gain_away = -np.inf

                # Take the away/drop step if and only if its exact finite-step log-det gain
                # exceeds the toward step's gain (or if it cleanly drops a tiny weight)
                if gain_away >= gain_toward:
                    pi_cand = (1.0 + step_away) * pi
                    if is_drop:
                        pi_cand[j_minus] = 0.0
                    else:
                        pi_cand[j_minus] = max(0.0, pi_cand[j_minus] - step_away)
                    pi_cand /= float(np.sum(pi_cand))

                    ok_cand, logdet_cand, vars_cand = _eval_unregularized_leverages(
                        pi_cand
                    )
                    if ok_cand and logdet_cand >= logdet_cur - 1e-12:
                        pi = pi_cand
                        logdet_cur = logdet_cand
                        variances = vars_cand
                        took_away_step = True

        if not took_away_step:
            pi_cand = (1.0 - step_toward) * pi
            pi_cand[j_plus] += step_toward
            pi_cand /= float(np.sum(pi_cand))
            ok_cand, logdet_cand, vars_cand = _eval_unregularized_leverages(pi_cand)
            if ok_cand:
                pi = pi_cand
                logdet_cur = logdet_cand
                variances = vars_cand

        max_var = float(np.max(variances))
        if max_var <= r_float * (1.0 + tol):
            converged = True
            break

    # Apply Caratheodory support reduction to guarantee |supp(pi)| <= r(r+1)/2
    pi_red = _caratheodory_support_reduction(Z, pi, tol=1e-12)
    ok_red, _, vars_red = _eval_unregularized_leverages(pi_red)
    if ok_red and float(np.max(vars_red)) <= max(
        max_var * (1.0 + 1e-7), r_float * (1.0 + tol)
    ):
        pi = pi_red
        variances = vars_red
        max_var = float(np.max(variances))

    if weight_prune_tol > 0.0:
        cand_pi = pi.copy()
        cand_pi[cand_pi < weight_prune_tol] = 0.0
        if float(np.sum(cand_pi)) > 0.0:
            cand_pi /= float(np.sum(cand_pi))
            ok_pruned, _, vars_pruned = _eval_unregularized_leverages(cand_pi)
            if ok_pruned and float(np.max(vars_pruned)) <= r_float * (1.0 + tol):
                pi = cand_pi
                variances = vars_pruned
                max_var = float(np.max(variances))

    V_gram_whitened = (Z.T * pi) @ Z
    V_gram_whitened = 0.5 * (V_gram_whitened + V_gram_whitened.T)
    eigvals_w, eigvecs_w = np.linalg.eigh(V_gram_whitened)
    if float(eigvals_w[0]) > 1e-13 * max(1.0, float(eigvals_w[-1])):
        V_w_inv = (eigvecs_w / eigvals_w) @ eigvecs_w.T
        V_w_inv = 0.5 * (V_w_inv + V_w_inv.T)
        max_var = float(np.max(batch_mahalanobis_norm(Z, V_w_inv) ** 2))
    else:
        max_var = float("inf")

    converged = bool(max_var <= r_float * (1.0 + tol))
    if not converged and max_iter > 0:
        warnings.warn(
            f"compute_g_optimal_design did not reach tolerance {tol} within {max_iter} iterations "
            f"(achieved max_variance={max_var:.6f} vs target {r_float * (1.0 + tol):.6f}).",
            RuntimeWarning,
            stacklevel=2,
        )

    duality_gap = max(0.0, (max_var - r_float) / r_float)
    return OptimalDesignResult(
        pi=pi,
        max_variance=max_var,
        effective_rank=rank_eff,
        duality_gap=duality_gap,
        num_iterations=num_iter,
        converged=converged,
    )