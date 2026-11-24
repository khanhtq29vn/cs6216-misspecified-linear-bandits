# Contextual and Linear Bandits under Model Misspecification

**CS6216 — Advanced Topics in Machine Learning, National University of Singapore · Group 13a**
Tran Quoc Khanh · Nurmukhammed Aitymbetov

This repository contains finite-action stochastic bandit implementations, three synthetic reward models, deterministic and regression tests, and a reproducible Monte Carlo benchmark. It accompanies our lecture notes on linear bandits under model misspecification.

## Model and algorithms

At each round, the expected reward of feature vector $x$ is

$$f_0(x)=\langle x,\theta^*\rangle+\varepsilon(x),$$

with additive Gaussian feedback noise in the simulations. Pseudo-regret is accumulated relative to the highest *true* expected reward in the current fixed action set.

| Implementation | Description |
| --- | --- |
| **Standard UCB** | Unstructured $k$-armed UCB, using each arm's own empirical reward mean. |
| **LinUCB (scaled OFUL)** | Shared-parameter ridge regression with an optimistic exploration bonus; the simulation uses `confidence_scale=0.60`. |
| **Basis-test RLB variant** | Tests linear consistency on a selected basis and up to three diagnostic arms, then runs a linear learner or unstructured UCB. |
| **Design-based PE (phase-local)** | Recomputes an approximate $G$-optimal allocation on each active set, interleaves support-arm pulls in feature-based order, estimates rewards using that phase's observations, and applies elimination. |

The implementations are experimental policies, not exact replicas of all theorem-calibrated algorithms in the cited papers. In particular, a fixed-basis RLB screen does not satisfy the general coverage assumption of Ghosh et al.'s Algorithm 1, and the empirical PE phase schedule and elimination multiplier `2.0` differ from Algorithm 2 in Liu et al. Merely changing its multiplier to `16` would not make the implementation theorem-equivalent.

## Experimental regimes

1. **Realizable:** $\varepsilon(x)=0$ for every arm; $k=50$, $d=5$ in the main benchmark.
2. **Sparse corruption:** one arm receives an additive perturbation of `0.45`; $k=20$, $d=5$ in the main benchmark. The geometry illustrates how a truly optimal corrupted arm can be overlooked by shared linear optimism.
3. **Pure $\rho$-gap-adjusted misspecification:** the environment checks $|\varepsilon(x)|\leq\rho(f^*-f_0(x))$ for every arm; $\rho=0.30$, $k=40$, $d=5$ in the main benchmark. The standard parameter sweep also enforces a maximum expected-reward gap of at most one.

These hand-constructed synthetic instances illustrate distinct mechanisms. They are not a representative sample of all bandit instances. Moreover, $\rho=0.30$ is outside the sufficient condition $\rho\leq 1/(16\sqrt d)$ for the modified PE algorithm analyzed by Liu et al.; its theorem cannot be inferred from the observed performance of the empirical PE variant.

## Reproducing the experiments

From the repository root, with Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_setup.py
```

A quick pipeline smoke test (isolated outputs, not a substitute for the full Monte Carlo experiment):

```bash
python -m experiments.run_experiments --quick \
  --output-dir experiments/quick/smoke
MPLBACKEND=Agg python -m experiments.plot_results \
  --input experiments/quick/smoke/results_summary.npz \
  --output-dir figures/quick/smoke
```

To regenerate the **full benchmark** with the prespecified design (20 seeds and 6,000 rounds in each main comparison; 15 seeds and 4,000 rounds in the parameter sweeps):

```bash
python -m experiments.run_experiments \
  --seeds-main 20 --seeds-sweep 15 \
  --horizon-main 6000 --horizon-sweep 4000 \
  --output-dir experiments/runs/reproduction
MPLBACKEND=Agg python -m experiments.plot_results \
  --input experiments/runs/reproduction/results_summary.npz \
  --output-dir figures/runs/reproduction
```

The runner refuses to overwrite existing outputs unless `--overwrite` is explicitly specified. The parameter sweeps pair environment/noise seeds across parameter values *within a policy*; noise streams differ across policies. Outputs include raw trial-level regret trajectories, rejection flags, summary statistics and source/runtime metadata. Exact floating-point trajectories can vary between runtimes near action-selection ties; the recorded benchmark used Python `3.15.0` and NumPy `2.5.4`.

## Recorded full benchmark

The source commit for the benchmark below is `e0f9c4f` (source fingerprint `ab98055b5090fd2d0454a58b85b92c56f37f09bf4985f8183db97598d551d1ba`). Values are **mean cumulative pseudo-regret ± standard error** over 20 Monte Carlo trials, at $n=6000$.

| Algorithm | Realizable | Sparse corruption | $\rho$-GAM ($\rho=0.30$) |
| --- | ---: | ---: | ---: |
| Standard UCB | 174.7 ± 3.4 | 44.0 ± 0.9 | 111.4 ± 1.7 |
| LinUCB (scaled OFUL) | 46.9 ± 1.2 | 1432.6 ± 0.7 | 25.0 ± 0.8 |
| Basis-test RLB variant | 60.6 ± 1.0 | 103.2 ± 0.7 | 141.5 ± 8.9 |
| Design-based PE (phase-local) | 1260.5 ± 9.7 | 1657.3 ± 76.2 | 898.8 ± 42.5 |

RLB rejected its diagnostic null hypothesis on **0/20**, **20/20**, and **16/20** trials, respectively. The substantial PE regret, including in the realizable instance, is part of the recorded outcome. Its empirical sample schedule and exploration costs should not be conflated with the guarantees of a differently calibrated PE algorithm.

The exact results, generated LaTeX table, plots and a file-hash manifest are preserved in [`experiments/archive/2026-11-24_interleaved_e0f9c4f/`](experiments/archive/2026-11-24_interleaved_e0f9c4f/). Figures: [realizable](experiments/archive/2026-11-24_interleaved_e0f9c4f/figures/fig1_realizable_benchmark.pdf), [sparse corruption](experiments/archive/2026-11-24_interleaved_e0f9c4f/figures/fig2_ghosh_sparse_misspec.pdf), [$\rho$-GAM](experiments/archive/2026-11-24_interleaved_e0f9c4f/figures/fig3_liu_gam_benchmark.pdf), and [structural diagnostics](experiments/archive/2026-11-24_interleaved_e0f9c4f/figures/fig4_structural_diagnostics.pdf). Plot bands and sweep error bars display **one standard error**, not 95% confidence intervals. The arm-count sweep compares constructed instances for each $k$, so it does not isolate action count from geometry.

Prior numerical artifacts are retained separately under [`experiments/archive/2026-10-29/`](experiments/archive/2026-10-29/) and [`experiments/archive/2026-11-24_pre_order/`](experiments/archive/2026-11-24_pre_order/); neither should be treated as outputs of the final interleaved PE implementation.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/` | Policies, environments and optimal-design utilities |
| `experiments/` | Monte Carlo runners, provenance, table and figure generation |
| `experiments/archive/` | Versioned historical and final numerical artifacts |
| `tests/` | Unit and research-integrity regression tests |
| `docs/` | Mathematical contracts and change history |
| `.github/workflows/` | Automated tests and pipeline smoke checks |
| `verify_setup.py` | Deterministic environment/algorithm smoke evaluation |

## References

- Li, Chu, Langford & Schapire (2010), [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146).
- Abbasi-Yadkori, Pál & Szepesvári (2011), [Improved Algorithms for Linear Stochastic Bandits](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html).
- Lattimore & Szepesvári (2020), [Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf).
- Lattimore, Szepesvári & Weisz (2020), [Learning with Good Feature Representations in Bandits and in RL with a Generative Model](https://proceedings.mlr.press/v119/lattimore20a/lattimore20a.pdf).
- Ghosh, Chowdhury & Gopalan (2017), [Misspecified Linear Bandits](https://arxiv.org/abs/1704.06880).
- Liu, Qiao, Yin, Bogunovic & Wang (2025), [No-Regret Linear Bandits under Gap-Adjusted Misspecification](https://arxiv.org/abs/2501.05361).

Software citation metadata is in [`CITATION.cff`](CITATION.cff). Code is distributed under the [MIT License](LICENSE).
