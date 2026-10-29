# Contextual and Linear Bandits under Model Misspecification

**Course:** CS6216 – Sequential Decision Making (National University of Singapore)  
**Instructor:** Prof. Jonathan Scarlett  
**Group 13a:** Tran Quoc Khanh & Nurmukhammed Aitymbetov  

---

## 1. Overview

This repository contains the reproducible Python simulation framework, mathematical unit test suite, Monte Carlo experimental benchmarks, and LaTeX lecture notes (`notes/main.tex` and `notes/slides.tex`) for **Group 13a**'s project on **Contextual and Linear Bandits under Model Misspecification**.

We investigate both theoretically and empirically how stochastic linear bandit algorithms behave when the classical linear realizability assumption $f_0(\mathbf{x}) = \langle \mathbf{x}, \boldsymbol{\theta}^* \rangle$ is violated across three distinct regimes:

1. **Regime 1 — Exact Realizability ($\varepsilon \equiv 0$, Section 2):**  
   Classical stochastic linear bandits where **LinUCB / OFUL** achieves $\mathcal{O}(d\sqrt{n}\log n)$ cumulative regret and **$G$-Optimal Phased Elimination** achieves $\mathcal{O}(\sqrt{dn\log(kn)})$, outperforming unstructured **Standard UCB** ($\mathcal{O}(\sqrt{kn\log n})$) when $d \ll k$ ([Li et al., 2010](https://arxiv.org/abs/1003.0146); [Abbasi-Yadkori et al., 2011](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html)).

2. **Regime 2 — Uniform & Sparse Subspace Misspecification (Section 3, [Ghosh et al., AAAI 2017](https://arxiv.org/abs/1704.06880)):**  
   When the true reward vector satisfies $\boldsymbol{\mu} = X^T \boldsymbol{\theta}^* + \boldsymbol{\varepsilon}$ with a positive sparse deviation $\varepsilon_k > \theta_1^* - \mu_k^{\mathrm{lin}}$ on an optimal arm whose feature vector lies in the span of uncorrupted suboptimal basis arms with shrinkage coordinates ($\alpha_1 + \alpha_2 < 1$), unmodified **LinUCB/OFUL** suffers catastrophic linear regret $\Omega(n)$ due to subspace extrapolation bias (Theorem 3.2). We implement the two-phase **Robust Linear Bandit (RLB)** algorithm (Algorithm 3), which tests the parameter-free subspace discrepancy $\langle \mathbf{c}, \hat{\boldsymbol{\mu}}_{\mathcal{B}} \rangle - \hat{\mu}_{d+1}$ in Phase 1 to safely interpolate between **LinUCB** ($\tilde{\mathcal{O}}(d\sqrt{n})$ under $H_0$) and **Standard UCB** ($\tilde{\mathcal{O}}(\sqrt{kn})$ under $H_1$, Theorem 3.3).

3. **Regime 3 — $\rho$-Gap-Adjusted Misspecification ($\rho$-GAM, Section 4, [Liu, Qiao, Yin, Bogunovic, & Wang, 2025](https://arxiv.org/abs/2501.05361)):**  
   When the pointwise linear approximation error is bounded proportionally to the true suboptimality gap, $|f_{\mathbf{w}^*}(\mathbf{x}) - f_0(\mathbf{x})| \le \rho (f^* - f_0(\mathbf{x}))$ for $\rho \in [0, 1)$ (Definition 4.1), the misspecification bias becomes self-bounding as suboptimal arms are eliminated. Sublinear no-regret guarantees are restored for both unmodified **LinUCB** ($\rho \lesssim \frac{1}{d\sqrt{\log n}}$, Theorem 4.3) and **$G$-Optimal Phased Elimination** ($\rho < \frac{1}{2\sqrt{d}+1} = \mathcal{O}(1/\sqrt{d})$, Theorem 4.4), even when the global uniform error $\|\boldsymbol{\varepsilon}\|_\infty$ on far-from-optimal arms is large.

---

## 2. Repository Architecture

```text
cs6216-misspecified-linear-bandits/
├── README.md                      # Project documentation and reproducibility guide
├── requirements.txt               # Python scientific computing dependencies
├── verify_setup.py                # Fast end-to-end verification across all 3 regimes
├── src/
│   ├── __init__.py                # Package exports
│   ├── utils.py                   # Sherman-Morrison updates, Pivoted-QR basis & Frank-Wolfe G-optimal design
│   ├── environments.py            # Self-verifying Realizable, Ghosh Sparse, and Liu rho-GAM environments
│   └── algorithms.py              # StandardUCB, LinUCB, RobustLinearBanditRLB, PhasedEliminationBandit
├── tests/
│   └── test_bandits.py            # Automated unittest suite verifying all mathematical lemmas & theorems
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py         # Multi-seed Monte Carlo simulation runner & LaTeX table exporter
│   └── plot_results.py            # Publication-quality vector PDF figure generator
├── figures/                       # Generated vector PDF plots included in `main.tex` and `slides.tex`
└── notes/
    ├── main.tex                   # CS6216 Lecture Notes LaTeX source (8-12 pages)
    └── slides.tex                 # CS6216 Presentation Slides LaTeX Beamer source (16:9)
```

---

## 3. Reproducibility & Execution Commands

### 3.1. Environment Setup (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2. Run Mathematical Unit Tests & Fast Verification
```bash
python -m unittest discover -s tests -v
python verify_setup.py
```

### 3.3. Run Full 20-Seed Monte Carlo Benchmarks & Export Figures
```bash
python -m experiments.run_experiments
python -m experiments.plot_results
```