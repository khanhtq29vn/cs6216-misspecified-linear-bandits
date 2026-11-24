"""
Vector PDF/PNG figure generator for the CS6216 synthetic benchmarks.

Loads `experiments/results_summary.npz` and exports four 2-panel figures to `figures/`:
1. `figures/fig1_realizable_benchmark.pdf`:
   (a) Cumulative regret R_t in the realizable linear regime (eps = 0).
   (b) Final regret R_n vs. number of arms k in {10, ..., 200} at fixed d = 5.
2. `figures/fig2_ghosh_sparse_misspec.pdf`:
   (a) Cumulative regret R_t under sparse subspace misspecification (eps_k = 0.45).
   (b) Phase transition of R_n and RLB hypothesis-test rejection rate across eps_k.
3. `figures/fig3_liu_gam_benchmark.pdf`:
   (a) Cumulative regret R_t under rho-GAM (rho = 0.30).
   (b) Final regret R_n across misspecification parameter rho in [0.05, 0.60].
4. `figures/fig4_structural_diagnostics.pdf`:
   (a) Geometric comparison of the rho-GAM cone vs. uniform l_inf tube.
   (b) Subspace extrapolation collapse of LinUCB UCB_k(t) in Ghosh et al.'s setting.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.provenance import verify_provenance


FIGURE_OUTPUT_DIR = "figures"

ALG_STYLES: Dict[str, Dict[str, object]] = {
    "Standard UCB": {
        "color": "#7F7F7F",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.1,
        "label": "Standard UCB (Unstructured)",
    },
    "LinUCB (scaled OFUL)": {
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.3,
        "label": "LinUCB (scaled OFUL)",
    },
    "Basis-test RLB variant": {
        "color": "#0072B2",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 2.3,
        "label": "Basis-test RLB variant",
    },
    "Design-based PE (phase-local)": {
        "color": "#009E73",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.3,
        "label": "Design-based PE (phase-local)",
    },
}


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "figure.titlesize": 11.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_shared_top_legend(fig: plt.Figure) -> None:
    handles: List[Line2D] = []
    labels: List[str] = []
    for style in ALG_STYLES.values():
        h = Line2D(
            [0],
            [0],
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            marker=str(style["marker"]),
            markersize=5.5,
            linewidth=float(style["linewidth"]),
        )
        handles.append(h)
        labels.append(str(style["label"]))

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#D0D7DE",
        fontsize=8.6,
        handlelength=2.4,
        columnspacing=1.4,
    )


def save_figure(fig: plt.Figure, base_name: str, out_dir: str | None = None) -> None:
    out_dir = out_dir or FIGURE_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    png_path = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  -> Exported {pdf_path} (and 300-DPI PNG preview)")


def plot_figure1_realizable(data: np.lib.npyio.NpzFile) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1))
    t_grid = data["t_grid"]

    max_y1 = 0.0
    for name, style in ALG_STYLES.items():
        mean = data[f"exp1_traj_{name}_mean"]
        se = data[f"exp1_traj_{name}_se"]
        max_y1 = max(max_y1, float(np.max(mean + se)))
        ax1.plot(
            t_grid,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax1.fill_between(
            t_grid, mean - se, mean + se, color=style["color"], alpha=0.16
        )

    ax1.set_title("(a) Realizable Linear Bandit ($\\varepsilon = 0,\\; k=50,\\; d=5$)")
    ax1.set_xlabel("Round $t$")
    ax1.set_ylabel("Cumulative Pseudo-Regret $R_t$")
    ax1.set_xlim(0, t_grid[-1])
    ax1.set_ylim(0, max_y1 * 1.12)

    k_grid = data["exp1_k_grid"]
    max_y2 = 0.0
    for name, style in ALG_STYLES.items():
        mean_k = data[f"exp1_ksweep_{name}_mean"]
        se_k = data[f"exp1_ksweep_{name}_se"]
        max_y2 = max(max_y2, float(np.max(mean_k + se_k)))
        ax2.errorbar(
            k_grid,
            mean_k,
            yerr=se_k,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=style["linewidth"],
            capsize=3,
        )

    ax2.set_title(
        f"(b) Regret vs. Number of Arms $k$ ($d=5$, $n={int(data['meta_sweep_horizon'][0]):,}$)"
    )
    ax2.set_xlabel("Number of Actions $k$")
    ax2.set_ylabel("Final Cumulative Regret $R_n$")
    ax2.set_xlim(0, 210)
    ax2.set_ylim(0, max_y2 * 1.38)

    ucb_k150 = float(data["exp1_ksweep_Standard UCB_mean"][-2])
    ax2.annotate(
        "Observed effect of action count $k$\n"
        "depends on features and policy",
        xy=(150, ucb_k150),
        xytext=(14, max_y2 * 0.84),
        arrowprops=dict(
            arrowstyle="->",
            color="#555555",
            lw=1.1,
            connectionstyle="arc3,rad=-0.12",
        ),
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8F9FA", ec="#CCCCCC", alpha=0.95),
    )

    add_shared_top_legend(fig)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    save_figure(fig, "fig1_realizable_benchmark")


def plot_figure2_ghosh_sparse(data: np.lib.npyio.NpzFile) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1))
    t_grid = data["t_grid"]

    max_y1 = 0.0
    for name, style in ALG_STYLES.items():
        mean = data[f"exp2_traj_{name}_mean"]
        se = data[f"exp2_traj_{name}_se"]
        max_y1 = max(max_y1, float(np.max(mean + se)))
        ax1.plot(
            t_grid,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax1.fill_between(
            t_grid, mean - se, mean + se, color=style["color"], alpha=0.16
        )

    if t_grid[-1] >= 160:
        ax1.axvline(160, color="#0072B2", linestyle=":", linewidth=1.5)
        ax1.text(
            0.04, 0.88,
            "RLB basis screen ($t=160$)\n"
            "Selection depends on diagnostic results",
            transform=ax1.transAxes, ha="left", va="top",
            color="#0072B2", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#EBF5FB", ec="#0072B2", alpha=0.92),
        )
    ax1.set_title(
        "(a) Sparse Misspecification ($\\varepsilon_k = 0.45,\\; k=20,\\; d=5$)"
    )
    ax1.set_xlabel("Round $t$")
    ax1.set_ylabel("Cumulative Pseudo-Regret $R_t$")
    ax1.set_xlim(0, t_grid[-1])
    ax1.set_ylim(0, max_y1 * 1.12)

    eps_grid = data["exp2_eps_grid"]
    span_patch = ax2.axvspan(
        0.215,
        0.52,
        color="#FADBD8",
        alpha=0.30,
        label="Corrupted Arm Optimal ($\\varepsilon_k > 0.215$)",
    )
    max_y2 = 0.0
    for name, style in ALG_STYLES.items():
        mean_e = data[f"exp2_sweep_{name}_mean"]
        se_e = data[f"exp2_sweep_{name}_se"]
        max_y2 = max(max_y2, float(np.max(mean_e + se_e)))
        ax2.errorbar(
            eps_grid,
            mean_e,
            yerr=se_e,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=style["linewidth"],
            capsize=3,
        )

    ax2.set_title(
        f"(b) Sparse Perturbation Sweep ($n={int(data['meta_sweep_horizon'][0]):,}$)"
    )
    ax2.set_xlabel("Sparse Deviation Magnitude $\\varepsilon_k$ on Corrupted Arm")
    ax2.set_ylabel("Final Cumulative Regret $R_n$")
    ax2.set_xlim(-0.01, 0.515)
    ax2.set_ylim(0, max_y2 * 1.24)

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    ax2_twin.spines["right"].set_visible(True)
    (rej_line,) = ax2_twin.plot(
        eps_grid,
        data["exp2_rlb_reject_rate"] * 100.0,
        color="#2C3E50",
        linestyle=":",
        marker="x",
        markersize=6.0,
        linewidth=1.7,
        label="RLB $H_0$ Rejection Rate (%)",
    )
    ax2_twin.set_ylabel("RLB $H_0$ Rejection Rate (%)", color="#2C3E50")
    ax2_twin.set_ylim(-5, 118)
    ax2_twin.grid(False)

    ax2.legend(
        handles=[span_patch, rej_line],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=8.1,
    )

    add_shared_top_legend(fig)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    save_figure(fig, "fig2_ghosh_sparse_misspec")


def plot_figure3_liu_gam(data: np.lib.npyio.NpzFile) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1))
    t_grid = data["t_grid"]

    max_y1 = 0.0
    for name, style in ALG_STYLES.items():
        mean = data[f"exp3_traj_{name}_mean"]
        se = data[f"exp3_traj_{name}_se"]
        max_y1 = max(max_y1, float(np.max(mean + se)))
        ax1.plot(
            t_grid,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax1.fill_between(
            t_grid, mean - se, mean + se, color=style["color"], alpha=0.16
        )

    ax1.set_title(
        f"(a) Gap-Adjusted Misspecification ($\\rho=0.30$, "
        f"$\\|\\boldsymbol{{\\varepsilon}}\\|_\\infty={float(data['diag_eps_inf'][0]):.3f}$)"
    )
    ax1.set_xlabel("Round $t$")
    ax1.set_ylabel("Cumulative Pseudo-Regret $R_t$")
    ax1.set_xlim(0, t_grid[-1])
    ax1.set_ylim(0, max_y1 * 1.14)

    rho_grid = data["exp3_rho_grid"]
    max_y2 = 0.0
    for name, style in ALG_STYLES.items():
        mean_r = data[f"exp3_sweep_{name}_mean"]
        se_r = data[f"exp3_sweep_{name}_se"]
        max_y2 = max(max_y2, float(np.max(mean_r + se_r)))
        ax2.errorbar(
            rho_grid,
            mean_r,
            yerr=se_r,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=style["linewidth"],
            capsize=3,
        )

    ax2.set_title(
        f"(b) Gap-Adjusted Sweep ($n={int(data['meta_sweep_horizon'][0]):,}$, $k=40$, $d=5$)"
    )
    ax2.set_xlabel("Gap-Adjusted Misspecification Parameter $\\rho$")
    ax2.set_ylabel("Final Cumulative Regret $R_n$")
    ax2.set_xlim(0.02, 0.63)
    ax2.set_ylim(0, max_y2 * 1.38)

    pe_val = float(
        data["exp3_sweep_Design-based PE (phase-local)_mean"][4]
    )
    ax2.annotate(
        "Empirical regret at fixed horizon;\n"
        "not an asymptotic guarantee",
        xy=(float(rho_grid[4]), pe_val),
        xytext=(0.04, max_y2 * 1.15),
        arrowprops=dict(
            arrowstyle="->",
            color="#009E73",
            lw=1.2,
            connectionstyle="arc3,rad=-0.12",
        ),
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", fc="#E8F8F5", ec="#009E73", alpha=0.95),
    )

    add_shared_top_legend(fig)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    save_figure(fig, "fig3_liu_gam_benchmark")


def plot_figure4_structural_diagnostics(data: np.lib.npyio.NpzFile) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    lin_val_grid = data["diag_lin_val_grid"]
    upper_cone = data["diag_gam_upper_cone"]
    lower_cone = data["diag_gam_lower_cone"]
    lin_sorted = data["diag_lin_sorted"]
    true_sorted = data["diag_true_sorted"]
    eps_inf = float(data["diag_eps_inf"][0])

    ax1.fill_between(
        lin_val_grid,
        lower_cone,
        upper_cone,
        color="#009E73",
        alpha=0.18,
        label="$\\rho$-GAM Cone ($|f_{\\mathbf{w}^*}-f_0| \\leq \\rho(f^*-f_0)$)",
    )
    ax1.plot(
        lin_val_grid,
        lin_val_grid,
        color="#2C3E50",
        linestyle="-",
        linewidth=1.6,
        label="Linear Surrogate $f_{\\mathbf{w}^*}(\\mathbf{x}) = \\langle \\mathbf{x}, \\mathbf{w}^* \\rangle$",
    )
    ax1.plot(
        lin_val_grid,
        lin_val_grid + eps_inf,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=f"Uniform $\\ell_\\infty$ Tube ($\\pm \\|\\boldsymbol{{\\varepsilon}}\\|_\\infty = \\pm {eps_inf:.2f}$)",
    )
    ax1.plot(
        lin_val_grid,
        lin_val_grid - eps_inf,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
    )
    ax1.scatter(
        lin_sorted[:-1],
        true_sorted[:-1],
        color="#0072B2",
        s=26,
        alpha=0.85,
        zorder=4,
        label="Suboptimal Arms ($i \\neq i^*$)",
    )
    ax1.scatter(
        [lin_sorted[-1]],
        [true_sorted[-1]],
        color="#D55E00",
        marker="*",
        s=155,
        zorder=5,
        label="Optimal Arm $\\mathbf{x}^*$ ($\\varepsilon(\\mathbf{x}^*) = 0$)",
    )

    ax1.set_title("(a) Geometry of $\\rho$-GAM Cone vs. Uniform $\\ell_\\infty$ Tube")
    ax1.set_xlabel(
        "Linear Prediction $f_{\\mathbf{w}^*}(\\mathbf{x}_i) = \\langle \\mathbf{x}_i, \\mathbf{w}^* \\rangle$"
    )
    ax1.set_ylabel("True Expected Reward $f_0(\\mathbf{x}_i)$")
    ax1.set_xlim(-0.56, 1.08)
    ax1.set_ylim(-0.85, 1.72)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.9)

    t_trace = data["diag_t_trace"]
    ucb_arm0 = data["diag_ucb_arm0"]
    ucb_opt = data["diag_ucb_opt"]
    pred_opt = data["diag_pred_opt"]
    true_opt = float(data["diag_true_opt_mean"][0])
    true_arm0 = float(data["diag_true_arm0_mean"][0])
    extrap_opt = float(data["diag_extrap_opt_mean"][0])

    ax2.axhline(
        true_opt,
        color="#009E73",
        linestyle="-",
        linewidth=1.8,
        label=f"True Optimal $f_0(\\mathbf{{x}}_k) = {true_opt:.3f}$",
    )
    ax2.axhline(
        true_arm0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label=f"Basis Arm 1 $\\mu_1 = {true_arm0:.2f}$",
    )
    ax2.axhline(
        extrap_opt,
        color="#7F7F7F",
        linestyle=":",
        linewidth=1.5,
        label=f"Extrapolated $\\mu_k^{{\\mathrm{{lin}}}} = {extrap_opt:.3f}$",
    )
    ax2.plot(
        t_trace,
        ucb_arm0,
        color="#D55E00",
        alpha=0.50,
        linewidth=1.3,
        label="LinUCB $\\mathrm{UCB}_1(t)$",
    )
    ax2.plot(
        t_trace,
        ucb_opt,
        color="#0072B2",
        linewidth=2.1,
        label="LinUCB $\\mathrm{UCB}_k(t)$ (Opt Arm)",
    )
    ax2.plot(
        t_trace,
        pred_opt,
        color="#0072B2",
        linestyle="-.",
        linewidth=1.2,
        alpha=0.80,
        label="Ridge Est. $\\langle \\mathbf{x}_k, \\hat{\\boldsymbol{\\theta}}_t \\rangle$",
    )

    ax2.set_title(
        "(b) Sparse Corruption: Linear Optimism Can Miss the Best Arm"
    )
    ax2.set_xlabel("Round $t$")
    ax2.set_ylabel("Reward / Upper Confidence Index")
    ax2.set_xlim(1, len(t_trace))
    ax2.set_ylim(0.24, 1.58)

    # Report a downward confidence-index crossing only if the recorded trace
    # actually contains one. Never invent a crossing for an illustrative plot.
    downward_crossings = np.flatnonzero(
        (ucb_opt[:-1] >= true_arm0) & (ucb_opt[1:] < true_arm0)
    )
    if downward_crossings.size > 0:
        cross_idx = int(downward_crossings[0]) + 1
        cross_round = int(t_trace[cross_idx])
        cross_y = float(ucb_opt[cross_idx])
        ax2.scatter([cross_round], [cross_y], color="#D91E18", s=42, zorder=6)
        ax2.annotate(
            f"Observed $\\mathrm{{UCB}}_k(t) < \\mu_1$ ($t = {cross_round}$):\n"
            "Optimistic extrapolation falls below the basis-arm mean",
            xy=(cross_round, cross_y),
            xytext=(max(5, int(len(t_trace) * 0.24)), 0.27),
            arrowprops=dict(
                arrowstyle="->",
                color="#D91E18",
                lw=1.3,
                connectionstyle="arc3,rad=-0.15",
            ),
            fontsize=8.0,
            bbox=dict(
                boxstyle="round,pad=0.28", fc="#FDEDEC", ec="#D91E18", alpha=0.95
            ),
        )
    elif np.all(ucb_opt < true_arm0):
        ax2.text(
            0.49, 0.06, "Index remains below the basis-arm mean in this trace",
            transform=ax2.transAxes, fontsize=8.0, ha="center",
        )
    else:
        ax2.text(
            0.49, 0.06, "No downward crossing observed within this horizon",
            transform=ax2.transAxes, fontsize=8.0, ha="center",
        )

    ax2.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        fontsize=7.5,
        columnspacing=0.9,
    )

    fig.tight_layout()
    save_figure(fig, "fig4_structural_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark figures from a result archive")
    parser.add_argument("--input", type=Path, default=Path("experiments/results_summary.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly authorize replacing existing figure files")
    args = parser.parse_args()
    npz_path = str(args.input)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Could not find {npz_path}. Please run `python -m experiments.run_experiments` first."
        )

    figure_stems = (
        "fig1_realizable_benchmark", "fig2_ghosh_sparse_misspec",
        "fig3_liu_gam_benchmark", "fig4_structural_diagnostics",
    )
    existing = [
        args.output_dir / f"{stem}.{extension}"
        for stem in figure_stems for extension in ("pdf", "png")
        if (args.output_dir / f"{stem}.{extension}").exists()
    ]
    if existing and not args.overwrite:
        parser.error("Existing figures detected; use a new --output-dir or explicit --overwrite")
    configure_matplotlib_style()
    with np.load(npz_path, allow_pickle=False) as data:
        if ("meta_method_names" not in data
            or "meta_complete" not in data
            or int(data["meta_complete"][0]) != 1
            or set(data["meta_method_names"]) != set(ALG_STYLES)):
            raise ValueError("Input results are incompatible with this plotting version")
        try:
            verify_provenance({k: data[k] for k in ("meta_schema_version", "meta_seed_protocol", "meta_source_sha256") if k in data})
        except ValueError as exc:
            raise ValueError(f"Cannot plot unverified or obsolete experiment archive: {exc}") from exc
        global FIGURE_OUTPUT_DIR
        FIGURE_OUTPUT_DIR = str(args.output_dir)
        print(f"Generating vector figures in {FIGURE_OUTPUT_DIR}...")
        plot_figure1_realizable(data)
        plot_figure2_ghosh_sparse(data)
        plot_figure3_liu_gam(data)
        plot_figure4_structural_diagnostics(data)
    print("All four figures exported")


if __name__ == "__main__":
    main()