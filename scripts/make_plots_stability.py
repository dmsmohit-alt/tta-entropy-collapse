"""Phase 2 figures from a saved toy_stability run directory (reads CSVs only).

Usage:  python scripts/make_plots_stability.py --run-dir results/toy_stability/<hash>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

C = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#7a7a7a"]
INK, MUTED = "#222222", "#777777"


def style(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def fig_alpha(pop: pd.DataFrame, out: Path) -> None:
    """Population curvature alpha_J(lam) per separation; zero crossing = lam_critical."""
    d = pop[pop.mc_seed == pop.mc_seed.min()]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (sep, g) in enumerate(d.groupby("sep")):
        ax.plot(g.lam, g.alpha_J, marker="o", ms=3.5, lw=1.5, color=C[i], label=f"sep = {sep:g}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xlabel="lambda (weight of -H(p_bar))", ylabel="alpha_J  (curvature at the balanced point)",
           title="Balanced point is stable iff alpha_J > 0 (population, MC)", yscale="symlog", ylim=(-0.2, 0.3))
    ax.set_yscale("symlog", linthresh=1e-3)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig5_alpha_vs_lambda.png", dpi=150)
    plt.close(fig)


def fig_eta(stab: pd.DataFrame, out: Path) -> None:
    d = stab[stab.eta_star_numeric.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, (lam, g) in enumerate(d.groupby("lam")):
        axs[0].scatter(g.eta_star_sample, g.eta_star_numeric, s=22, color=C[i], label=f"lambda = {lam:g}")
    lim = [d.eta_star_sample.min() * 0.8, d.eta_star_sample.max() * 1.25]
    axs[0].plot(lim, lim, color=MUTED, lw=0.8, ls=":", label="y = x")
    axs[0].set(xscale="log", yscale="log", xlabel="eta* predicted, 2 / lambda_max(Hessian)",
               ylabel="eta* measured (nonlinear GD bisection)", title="Predicted vs measured eta*")
    s = stab[stab.locally_stable_for_some_eta]
    for i, f in enumerate([0.5, 0.9, 1.1, 1.5]):
        axs[1].scatter(np.full(len(s), f) + np.random.default_rng(0).uniform(-0.02, 0.02, len(s)),
                       s[f"rho_map_{f}"], s=14, color=C[i])
    axs[1].axhline(1.0, color=INK, lw=0.8)
    axs[1].set(xlabel="eta / eta*", ylabel="spectral radius of GD map  I - eta H", title="Stable iff radius < 1")
    for a in axs:
        style(a)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_eta_star.png", dpi=150)
    plt.close(fig)


def fig_threshold(pop: pd.DataFrame, roots: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    p0 = pop[pop.lam == 0.0]
    all_ok = p0.groupby("margin").g_c_status.agg(lambda x: (x == "ok").all())   # drop margins where MC seeds disagree
    g = p0[p0.margin.isin(all_ok[all_ok].index)].groupby("margin").g_c.mean().reset_index()
    ax.plot(g.margin, g.g_c, color=C[0], marker="o", ms=4, lw=1.5, label="population reduced-flow root (derived 1-D flow)")
    r = roots[roots.phase1_status == "ok"]
    ax.scatter(r.sep**2, r.phase1_delta_star, s=22, color=C[3], zorder=3, label="Phase 1 simulated delta* (per seed)")
    ax.plot([0, 26], [0, 26], color=MUTED, lw=0.8, ls=":", label="delta* = margin")
    ax.set(xlabel="logit margin sep^2 / sigma^2", ylabel="critical bias gap", title="Collapse threshold: theory-side root vs simulation")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig7_threshold_theory_vs_sim.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    run = Path(ap.parse_args().run_dir)
    out = run / "figures"
    out.mkdir(exist_ok=True)
    pop, stab = pd.read_csv(run / "population.csv"), pd.read_csv(run / "stability.csv")
    roots = pd.read_csv(run / "sample_roots.csv") if (run / "sample_roots.csv").stat().st_size > 5 else pd.DataFrame()
    fig_alpha(pop, out)
    fig_eta(stab, out)
    if len(roots):
        fig_threshold(pop, roots, out)
    print(out)


if __name__ == "__main__":
    main()
