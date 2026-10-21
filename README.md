# Contextual and Linear Bandits under Model Misspecification

**Course:** CS6216 – Sequential Decision Making (National University of Singapore)  
**Instructor:** Prof. Jonathan Scarlett  
**Group 13a:** Tran Quoc Khanh & Nurmukhammed Aitymbetov  

---

## 1. Overview

This repository contains the reproducible Python simulation framework, experimental benchmarks, and LaTeX lecture notes for **Group 13a**'s project on **Contextual and Linear Bandits under Model Misspecification**.

We investigate both theoretically and empirically how stochastic linear bandit algorithms behave when the classical linear realizability assumption $f_0(\mathbf{x}) = \langle \mathbf{x}, \mathbf{\theta}^* \rangle$ is violated across three distinct regimes:
1. **Exact Realizability ($\varepsilon = 0$):** Classical stochastic linear bandits where **LinUCB / OFUL** achieves $\tilde{\mathcal{O}}(d\sqrt{n})$ cumulative regret, outperforming unstructured **UCB** ($\tilde{\mathcal{O}}(\sqrt{kn})$) when $d \ll k$ ([Li et al., 2010](https://arxiv.org/abs/1003.0146); [Abbasi-Yadkori et al., 2011](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html)).
2. **Sparse / Uniform Model Misspecification ([Ghosh et al., AAAI 2017](https://arxiv.org/abs/1704.06880)):** When the true reward vector satisfies $\mathbf{\mu} = X^T \mathbf{\theta}^* + \mathbf{\varepsilon}$ with a large deviation on an optimal arm whose feature vector lies in the span of suboptimal basis arms, unmodified **LinUCB/OFUL** suffers catastrophic linear regret $\Omega(n)$ due to subspace extrapolation bias. We implement the **Robust Linear Bandit (RLB)** algorithm, which performs a Phase-1 linearity hypothesis test on $d+1$ arms to safely switch between **LinUCB** ($\tilde{\mathcal{O}}(d\sqrt{n})$ under $H_0$) and **Standard UCB** ($\tilde{\mathcal{O}}(\sqrt{kn})$ under $H_1$).
3. **$\rho$-Gap-Adjusted Misspecification ($\rho$-GAM, [Liu, Qiao, Yin, Bogunovic, & Wang, 2025](https://arxiv.org/abs/2501.05361)):** When the pointwise linear approximation error is bounded proportionally to the true suboptimality gap, $|f_{\mathbf{w}^*}(\mathbf{x}) - f_0(\mathbf{x})| \le \rho (f^* - f_0(\mathbf{x}))$ for $\rho \in [0, 1)$, sublinear $\tilde{\mathcal{O}}(d\sqrt{n})$ no-regret guarantees are restored for both **LinUCB** (for moderate $\rho$) and **$G$-Optimal Design Phased Elimination** (up to $\rho = \mathcal{O}(1/\sqrt{d})$), even when the uniform sup-norm misspecification $\|\mathbf{\varepsilon}\|_\infty$ is large.

---

## 2. Repository Structure

```text
cs6216-misspecified-linear-bandits/
├── README.md                      # Project documentation and reproducibility guide
├── requirements.txt               # Python package dependencies
├── verify_setup.py                # Fast sanity-check script for environments and algorithms
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Sherman-Morrison inverse updates & Frank-Wolfe G-optimal design
│   ├── environments.py            # Realizable, Ghosh Sparse Misspecified, and Liu rho-GAM environments
│   └── algorithms.py              # Standard UCB, LinUCB/OFUL, RLB (Ghosh 2017), Phased Elimination (Liu 2025)
├── experiments/
│   ├── run_experiments.py         # Multi-seed Monte Carlo simulation runner (Stage 2.3)
│   └── plot_results.py            # Publication-quality PDF figure generator (Stage 2.3)
├── figures/                       # Generated vector PDF plots included in the lecture notes
└── notes/
    └── main.tex                   # CS6216 Lecture Notes LaTeX source
```

---

## 3. Quick Start & Reproducibility

### 3.1. Environment Setup (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Verify Core Implementation
To run a fast verification of the Kiefer-Wolfowitz $G$-optimal design solver and all four algorithms across the three misspecification regimes:
```bash
python verify_setup.py
```

### 3.3. Run Full Multi-Seed Experiments and Generate Figures
```bash
python -m experiments.run_experiments
python -m experiments.plot_results
```