"""Make Phase 1 figures from a saved run directory (reads CSVs only; no recomputation).

Usage:  python scripts/make_plots.py --run-dir results/toy_basin/<hash>
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

# Fixed-order, colour-blind-safe categorical colours (Okabe-Ito subset); never cycled.
C = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
INK, MUTED = "#222222", "#777777"


def style(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def fig_trajectories(run: Path, out: Path, logK: float) -> None:
    f = run / "trajectories.csv"
    if not f.exists():
        return
    tr = pd.read_csv(f)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (r, g) in enumerate(tr.groupby("ratio")):
        kw = dict(color=C[i % len(C)], lw=1.6, ls="-" if r < 1 else "--", label=f"delta0 = {r:g} x delta*")
        axs[0].plot(g.step, g.H_bar, **kw)
        axs[1].plot(g.step, g.b_gap, **kw)
    axs[0].axhline(logK, color=MUTED, lw=0.8, ls=":")
    axs[0].set(xlabel="GD step", ylabel="H(p_bar)  [nats]", title="Marginal entropy")
    axs[1].set(xlabel="GD step", ylabel="b_0 - mean(b_others)", title="Bias of class 0 vs the rest")
    axs[0].text(axs[0].get_xlim()[1] * 0.98, logK, " log K", ha="right", va="bottom", color=MUTED, fontsize=8)
    for a in axs:
        style(a)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Entropy-minimisation GD, bias-only adaptation (solid: below delta*, dashed: above)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_trajectories.png", dpi=150)
    plt.close(fig)


def fig_basin_1d(sweep: pd.DataFrame, out: Path, sep: float, logK: float, tau: float) -> None:
    d = sweep[np.isclose(sweep.sep, sep)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (adapt, g) in enumerate(d.groupby("adapt")):
        s = g.groupby("delta").agg(m=("H_bar_final", "mean"), sd=("H_bar_final", lambda x: x.std(ddof=0)),
                                   c=("collapsed", "mean")).reset_index()
        label = {"b": "adapt b only", "Wb": "adapt W and b"}[adapt]
        axs[0].plot(s.delta, s.m, color=C[i], lw=1.6, marker="o", ms=3, label=label, ls="-" if adapt == "b" else "--")
        axs[0].fill_between(s.delta, np.clip(s.m - s.sd, 0, None), s.m + s.sd, color=C[i], alpha=0.15, lw=0)
        axs[1].plot(s.delta, s.c, color=C[i], lw=1.6, marker="o", ms=3, label=label, ls="-" if adapt == "b" else "--")
    margin = sep**2
    for a in axs:
        a.axvline(margin, color=MUTED, lw=0.8, ls=":", label="margin sep^2/sigma^2")
        style(a)
    axs[0].axhline(logK, color=MUTED, lw=0.8, ls=":")
    axs[0].axhline(tau * logK, color=MUTED, lw=0.8, ls="--")
    axs[0].text(0.2, tau * logK + 0.03, f"collapse level (tau={tau:g})", color=MUTED, fontsize=8)
    axs[0].set(xlabel="initial bias delta added to b_0 (logits)", ylabel="final H(p_bar)  [nats]",
               title="Final marginal entropy (mean +/- std over seeds)")
    axs[1].set(xlabel="initial bias delta added to b_0 (logits)", ylabel="fraction of seeds collapsed",
               title="Collapse yes/no", ylim=(-0.05, 1.05))
    axs[0].legend(frameon=False, fontsize=8, loc="center left")
    axs[1].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle(f"Basin boundary, sep = {sep:g}, K classes, {d.seed.nunique()} seeds", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_basin_1d.png", dpi=150)
    plt.close(fig)


def fig_basin_2d(sweep: pd.DataFrame, thr: pd.DataFrame, out: Path) -> None:
    d = sweep[sweep.adapt == "b"]
    d = d.assign(delta_rel=d.delta_rel.round(6))
    piv = d.groupby(["sep", "delta_rel"]).collapsed.mean().unstack("delta_rel")
    seps, rels = piv.index.values, piv.columns.values
    dr = np.diff(rels).mean()
    xe = np.append(rels - dr / 2, rels[-1] + dr / 2)
    ye = np.arange(len(seps) + 1) - 0.5
    fig, ax = plt.subplots(figsize=(7, 4.3))
    m = ax.pcolormesh(xe, ye, piv.values, cmap="Blues", vmin=0, vmax=1, shading="flat")
    ok = thr[(thr.budget == "default") & (thr.status == "ok")]
    for j, sp in enumerate(seps):
        g = ok[np.isclose(ok.sep, sp)]
        ax.scatter(g.delta_star / g.margin, [j] * len(g), color=INK, s=14, zorder=3,
                   label="bisected delta* (per seed)" if j == 0 else None)
    ax.set_yticks(range(len(seps)), [f"{s:g}" for s in seps])
    ax.set(xlabel="initial bias delta / margin  (margin = sep^2 / sigma^2)", ylabel="separation sep",
           title="Collapse fraction over seeds (dark = collapses), bias-only adaptation")
    ax.tick_params(labelsize=9)
    fig.colorbar(m, ax=ax, label="fraction collapsed")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out / "fig3_basin_2d.png", dpi=150)
    plt.close(fig)


def fig_threshold(thr: pd.DataFrame, out: Path) -> None:
    d = thr[thr.budget == "default"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ok = d[d.status == "ok"]
    z = d[d.status == "collapses_at_zero"]
    ax.scatter(ok.margin, ok.delta_star, color=C[0], s=22, label="delta* (per seed)")
    ax.scatter(z.margin, np.zeros(len(z)), facecolors="none", edgecolors=C[3], s=30,
               label="collapses already at delta = 0")
    mx = d.margin.max() * 1.05
    ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls=":", label="delta* = margin")
    ax.set(xlabel="logit margin sep^2 / sigma^2", ylabel="threshold delta*", title="Collapse threshold vs margin")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_threshold_vs_margin.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    run = Path(ap.parse_args().run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    out = run / "figures"
    out.mkdir(exist_ok=True)
    logK, tau, sep = float(np.log(cfg["toy"]["K"])), cfg["toy"]["collapse_tau"], cfg["toy"]["sep"]
    sweep, thr = pd.read_csv(run / "basin_sweep.csv"), pd.read_csv(run / "thresholds.csv")
    fig_trajectories(run, out, logK)
    fig_basin_1d(sweep, out, sep, logK, tau)
    fig_basin_2d(sweep, thr, out)
    fig_threshold(thr, out)
    print(out)


if __name__ == "__main__":
    main()
