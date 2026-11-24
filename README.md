**CS6216 · Sequential Decision Making · NUS · Group 13a**
Tran Quoc Khanh and Nurmukhammed Aitymbetov
This repository provides synthetic finite-action bandit environments, four numerical
policies, deterministic tests, and a multi-seed Monte Carlo evaluation pipeline.
The experiment implementations are distinguished from the theoretical algorithms
in the papers discussed in our lecture notes.
## Scope and mathematical model
An arm with feature vector `x` has expected reward
`f0(x) = x @ theta_star + epsilon(x)`.
The experiments compare:
| Policy name in generated output | Implementation |
| --- | --- |
| `Standard UCB` | Unstructured UCB with Gaussian/sub-Gaussian-width bonus |
| `LinUCB (scaled OFUL)` | Shared-feature ridge optimism with confidence multiplier `0.60` |
| `Basis-test RLB variant` | Fixed-basis linearity screen using up to three diagnostic arms, followed by an optimistic linear policy or unstructured UCB |
| `Design-based PE (phase-local)` | New G-optimal allocation in each phase, phase-local OLS, geometrically increasing phase budgets, and noise-calibrated elimination |
**Comparison with the literature.** The LinUCB confidence multiplier is tuned for
simulation; it is not the confidence radius in Theorem 9 of Liu et al. The
basis-screening policy is not Algorithm 1 of Ghosh et al.: it does not enforce
the non-sparse-deviation/coverage conditions needed for the original guarantee.
The PE implementation fits each phase independently, but its empirical sample
schedule and elimination threshold are not the theorem-calibrated Algorithm 2
of Liu et al. The latter uses a factor `16` with a particular confidence
convention and sample schedule. Changing only `elimination_multiplier` to `16`
does not make these code paths equivalent.
The synthetic regimes are:
1. **Realizable:** zero approximation error and a unique best arm.
2. **Sparse corruption:** one arm receives an additive reward perturbation,
   illustrating the hidden-optimum failure of optimistic linear extrapolation.
3. **Pure rho-GAM:** pointwise errors satisfy
   `abs(epsilon(x)) <= rho * (f_star - f0(x))`, checked arm-by-arm during
   construction. On the standard rho sweep `0 <= rho <= 0.60`, all expected
   reward gaps are at most `1`, matching the bounded-gap normalization used
   in Liu et al.'s Assumption 5.
The hand-constructed synthetic action sets are illustrative, not a random sample of
all realizable, sparse or gap-adjusted problems. Their parameter choices were
fixed in source before the new benchmark is run; one cannot infer universal
algorithm rankings from these instances. No settings should be selected after
examining the benchmark outputs without documenting a new experimental version.
The experiment's `rho = 0.30` is **outside** the sufficient theorem range
`rho <= 1/(16*sqrt(d))` for the modified PE in Liu et al. Finite-horizon
figures do not establish asymptotic regret results.
## Reproducible setup
Run commands from the repository root with Python 3.11 or later:
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_setup.py
An end-to-end **small smoke run** with isolated outputs:

python -m experiments.run_experiments --quick \
  --seeds-main 1 --seeds-sweep 1 \
  --horizon-main 30 --horizon-sweep 20 \
  --output-dir experiments/quick/smoke
MPLBACKEND=Agg python -m experiments.plot_results \
  --input experiments/quick/smoke/results_summary.npz \
  --output-dir figures/quick/smoke
Run the **full benchmark** (main: 20 seeds × 6,000 rounds;
secondary sweeps: 15 seeds × 4,000 rounds) and generate figures:

MPLBACKEND=Agg python -m experiments.plot_results
```

To rerun only the three main regimes (without auxiliary sweeps), use
`python -m experiments.run_main_only`. It writes a distinct, incomplete
archive in `experiments/main_only/`, and never masquerades as a full run.

The default full run writes `experiments/results_summary.npz`,
`experiments/table_results.tex`, and eight PDF/PNG figure files in `figures/`.
The `--quick` switch defaults to three seeds and short horizons
(600 main rounds, 400 sweep rounds), writes to `experiments/quick/`, and
does not overwrite full outputs. Command-line overrides allow reproducibility checks
with smaller seed counts and horizons; the output table reads these values
from the archive rather than hardcoding them. The `--only-exp3b` mode additionally checks the exact source SHA-256 digest,
seed protocol, version and experiment settings. It rejects historical archives
or archives from another source revision rather than mixing outputs. Existing
numerical outputs and figures are never silently replaced: use a fresh
`--output-dir` or an explicit `--overwrite` after preserving previous results.

**Timing:** The complete multi-seed, multi-sweep experiment can be costly on
laptops. Small smoke runs test pipeline integrity, not statistical precision.
The two main and sweep seed counts are intentionally different.

## Data lineage

The uploaded source archive contained a numerical result table, NPZ, and
figures generated by an earlier algorithm implementation. These are retained
without modification under `experiments/archive/2026-10-29/` and are **not**
outputs of the revised code. See
[`experiments/archive/2026-10-29/ORIGIN.md`](experiments/archive/2026-10-29/ORIGIN.md).

Fresh results are deliberately absent from the source package until the
corresponding benchmark command is executed. The archive contains metadata
for seed counts, horizons, method identifiers, Python/NumPy versions and Git
revision (when available). Individual noise streams use deterministic NumPy random generators with
documented seeds. The epsilon and rho sweeps use the **same trial seeds at
every parameter level**, so the feature geometry, arm permutation and noise
streams are paired; only the investigated model parameter changes. Different
algorithms currently use independent noise streams. The NPZ also stores raw
per-trial regret trajectories/final regret values and RLB rejection indicators
so every mean, standard error and parameter-level comparison can be audited
without re-running the experiment. The archive records a source SHA-256,
seed-protocol version, Git worktree state, and runtime versions; raw NPZ
files and command logs should be retained alongside any published results.

## Repository layout

```text
src/                 Policies, reward environments, optimal design utilities
experiments/         Runner, TeX table generator, figure generator
experiments/archive/ Historical benchmark outputs with provenance
figures/             Generated figures (populated after a run)
tests/               Unit and regression tests
.github/workflows/   Continuous integration smoke checks
docs/                Model contracts and change-impact register
verify_setup.py      Deterministic numerical smoke evaluation

## Primary literature

- Li, Chu, Langford & Schapire (2010), [Contextual-Bandit News Recommendation](https://arxiv.org/abs/1003.0146).
- Abbasi-Yadkori, Pál & Szepesvári (2011), [Improved Algorithms for Linear Stochastic Bandits](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html).
- Ghosh, Chowdhury & Gopalan (2017), [Misspecified Linear Bandits](https://arxiv.org/abs/1704.06880).
- Lattimore, Szepesvári & Weisz (2020), *Learning with Good Feature Representations in Bandits and in RL with a Generative Model*.
- Liu, Qiao, Yin, Bogunovic & Wang (2025), [No-Regret Linear Bandits under Gap-Adjusted Misspecification](https://arxiv.org/abs/2501.05361).

MIT license. See [`LICENSE`](LICENSE).
