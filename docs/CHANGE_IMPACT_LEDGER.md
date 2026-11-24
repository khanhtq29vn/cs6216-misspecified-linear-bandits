# Code-revision impact on the lecture notes

This release changes the simulation semantics relative to the source archive
uploaded on 2026-11-24. **Do not update lecture notes by replacing labels
while retaining the old numerical values.**

| Affected material | Before | Revised code | Required next action |
| --- | --- | --- | --- |
| PE | Cumulative cross-phase reward sufficient statistics; phase targets from all observations | Refit OLS using only current-phase observations and double design budgets | Rerun all main/sweep experiments and update PE rows and plots |
| PE theorem identity | Described as Liu et al., Algorithm 2 | Separately identified empirical phase-local variant | Do not attribute Liu's Theorem 16 to this implementation |
| RLB | Label suggested the full Ghosh RLB | Fixed-basis diagnostic variant with explicit coverage limit | Correct name in tables/figures |
| LinUCB | Often labelled OFUL | Explicitly scaled OFUL-style UCB (`0.60`) | Distinguish experimental settings from theorem-calibrated OFUL |
| Standard rho-GAM environment | Could violate the paper's bounded-gap normalization in the benchmark range | Rescaled surrogate and adjusted bias-sign allocation so max gap <=1 for rho <=0.60 | Recompute all rho-GAM numbers, diagnostics and figures; global epsilon magnitude changes |
| Quick-run artifact | Same output path and hardcoded 20-trial table caption | Separate output directory and metadata-driven caption | Run a fresh quick and full check |
| Figures | Annotated PE as universally sublinear based on finite trajectories | Describes finite-horizon results without unwarranted asymptotic inference | Regenerate all figures |
| Original `.npz`, `.tex`, figures | In active result paths | Preserved verbatim under archive | Keep historical provenance, do not mix with new outputs |

The archive's old Group 13a table (20 main seeds, horizon 6,000) lists
UCB `174.7, 44.0, 81.8`; LinUCB `46.9, 1432.6, 14.8`;
RLB `60.6, 103.2, 170.1`; PE `68.3, 812.1, 35.2`
(realizable, sparse, rho-GAM respectively). These values are **historical**;
none is claimed as a new result from the revised source.

A subsequent lecture-notes update must check Section 5's experimental
configuration and Table 1 as well as references to the previous `rho=0.30`
maximum approximation error `0.35` (revised benchmark has `0.2275`).

## Pre-release scientific-integrity corrections (24 November 2026)

- Epsilon and rho sweeps now reuse the same environment and feedback seed for
  every parameter level. Earlier sweep curves had a different randomly
  generated feature matrix at every level; they must not be interpreted as
  paired estimates of how regret changes with one parameter.
- Raw per-seed regret trajectories, sweep outcomes and RLB rejection flags are
  retained so that means and standard errors can be reconstructed.
- `--only-exp3b` refuses to mix sources with a different exact code fingerprint,
  seed protocol, NumPy/Python runtime, method identifiers, seed counts or
  horizons; outputs require an explicit overwrite flag before replacement.
- The source and metadata changes alter **the reproducibility protocol** and
  the parameter-sweep estimates. Neither the theory nor the experiment implies
  uniformly superior performance of any policy.

## Finite-horizon phase ordering correction (post-full-benchmark audit)

The full benchmark run archived under its original SHA-256 fingerprint exposed
an index-order artifact: the PE implementation drew every allocated sample of
one support arm before starting the next arm. Large late batches truncated by
`n=6000` therefore depended strongly on which arm happened to have the lowest
input index. A controlled experiment holding the physical arms and per-arm
noise observations fixed produced regrets of 1185.58 and 1430.20 solely from
arm relabeling.

The candidate fix interleaves already-planned samples in feature-based order.
No regret bound, sample budget, elimination threshold, reward geometry, seed,
or fitting formula is changed. New outputs must use a new source fingerprint
and a fresh full benchmark; the pre-correction NPZ and figure PDFs are
historical evidence and must not be presented as results from the candidate.

The source SHA-256 contract also now includes the figure-generation source
`experiments/plot_results.py`; the earlier hash did not. Generated figure PDFs
must be accompanied by an output artifact manifest if published.
