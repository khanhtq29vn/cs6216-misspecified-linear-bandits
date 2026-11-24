# Algorithm and numerical-evidence contracts

The finite-arm stochastic reward model is `Y_t = <x_t, theta> + eps(x_t) + eta_t`
with Gaussian feedback noise of specified standard deviation `sigma`.

## Policies

**UCB:** Pulls unobserved arms first; thereafter chooses the largest empirical
mean plus `sqrt(c_bonus*sigma**2*log(max(t,2))/N_i)`. A fixed simulated choice
of `c_bonus` is not a proof of the optimal exploration constant.

**Shared LinUCB:** Ridge regression with a Sherman–Morrison matrix inverse
update and a self-normalized-style radius computed from a log-determinant
upper bound. The experiment multiplies the radius by `0.60`; consequently the
exact-realizability OFUL confidence theorem does not apply directly.

**Basis-test RLB variant:** The test statistic for diagnostic arm `j` is
`D_j = c_j.T @ mu_basis - mu_j`. With `tau` samples per tested arm,
its Gaussian-variance proxy is
`sigma**2*(1+||c_j||**2)/tau`, and a union bound controls false rejection
*only on the tested diagnostics*. Acceptance does not certify untested arms;
see `test_unexamined_optimal_arm_can_pass_diagnostic_test`.

**Phase-local PE:** A fresh approximate G-optimal design `pi_l` is calculated
on each active set. Integer counts are the ceiling of `m_l*pi_l`. The current
phase's OLS estimate uses **only** that phase's feature/reward totals.
The Gram matrix is checked for invertibility on the active feature span.
A numerical certificate requires `max_x x.T @ M(pi_l)^+ @ x <= 2*r_l`.
The phase budget doubles and the elimination threshold uses
`elimination_multiplier * 2*sigma*sqrt(r_l*log(2*k/delta_l)/m_l)`.
The default empirical multiplier is `2`, and this exact schedule is distinct
from Liu et al. (2025), Algorithm 2, where the modified threshold is
`16*sqrt((d/m)*log(1/alpha))` under the paper's sampling conventions.

## Environment claims

- Realizable: `epsilon == 0` exactly up to machine rounding.
- Sparse: precisely one arm has nonzero `epsilon`; this is not the source
  paper's algorithm-independent impossibility construction.
- Pure rho-GAM: the environment verifies the pointwise bound and identical
  maximizers for the chosen surrogate `w_star`. For standard sweeps up to
  `rho = 0.60`, it additionally enforces `max_x Delta(x) <= 1`.
- The default `rho = 0.30` is larger than the sufficient PE theorem regime
  `rho <= 1/(16*sqrt(d))`. Neither algorithm's empirical output is a proof of
  a general no-regret bound.

## Evidentiary standard

Unit tests certify deterministic code properties and small numerical
identities. Monte Carlo means and standard errors summarize the seeded
benchmark outcomes; they do not certify high-probability regret inequalities.
All changed algorithms or reward geometries require new experiment outputs
before a table in the lecture notes is updated.

## Reproducible experimental comparison

For the epsilon and rho parameter sweeps, each Monte Carlo replication uses
an identical environment and noise seed at every parameter value. This is a
paired design within each policy, not a claim that the algorithms have identical
feedback histories. Algorithm-specific feedback generators remain independent.
The raw trial-level pseudo-regret and diagnostic rejection records are retained
alongside summary statistics for auditing and paired-difference analysis.

The source of this teaching benchmark is intentionally hand-constructed.
Performance on it alone is not a general ranking of bandit algorithms. In
particular, rho=0.30 lies outside the cited sufficient regret regime for the
modified PE. Do not choose or change the geometry, exploration constants,
seeds, or metric post hoc to improve figures. Changes require a new documented
experiment revision and a full re-run.

The archive's source fingerprint and explicit `--overwrite` gate prevent
accidental mixing/overwriting. Fingerprints establish file identity; they are
not signatures that prove a computation was executed or prevent malicious edits.
