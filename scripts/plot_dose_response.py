"""Error vs 'SGD time' eta x steps for runs that differ only in the learning rate (dose-response / gradient-flow check).

Why: for small plain-SGD steps the parameter trajectory depends on eta x (number of steps) ('gradient-flow time'). If degradation is a deterministic
drift, curves of different eta should lie on top of each other when plotted against t = eta x steps; if minibatch noise matters, smaller eta degrades
LESS at equal t (noise per unit t scales with eta). Large eta leaves the small-step regime, so deviations there are expected and are the point.
Cycle k is placed at its mid-point: t_k = eta * steps_per_cycle * (k + 0.5)  (a cycle average mixes 15 corruptions: coarse by construction).

Usage (reads saved runs; run it on the machine that has results/):
  python scripts/plot_dose_response.py --exp-dir results/stream_tent --seeds 0 \
      --methods tent_im_sgd_lam1 tent_im_sgd_lam1_c0p9 tent_im_sgd_lam1_c1p1 tent_im_sgd_lam1_c2p0 [--bn-ref bn_adapt]
Use --seeds 0 to compare runs that exist for seed 0 only (otherwise a 3-seed mean is compared with a single seed: a confound).
The first method in --methods is the reference for the 'minus reference' table. Writes dose_response.png / dose_response_table.csv
into --out (default: <exp-dir>/dose_response). Linear interpolation, NO extrapolation (NaN outside a curve's t range).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_runs as agg  # noqa: E402


def t_axis(lr: float, steps_per_cycle: int, n_cycles: int) -> np.ndarray:
    return lr * steps_per_cycle * (np.arange(n_cycles) + 0.5)


def interp_table(curves: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray, ref: str) -> pd.DataFrame:
    """curves: name -> (t, err). Returns rows = t grid, columns = err per curve (NaN outside its range) and '<name> - <ref>'."""
    df = pd.DataFrame({"t": grid})
    for name, (t, e) in curves.items():
        df[name] = np.interp(grid, t, e, left=np.nan, right=np.nan)
    for name in curves:
        if name != ref:
            df[f"{name} - {ref}"] = df[name] - df[ref]
    return df


def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], lrs: dict[str, float], refs: dict[str, float], out_png: Path,
                title_note: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    names = sorted(curves, key=lambda n: lrs[n])                 # magnitude (lr) -> ONE hue, light -> dark (sequential)
    cols = [cm.Blues(0.35 + 0.6 * i / max(1, len(names) - 1)) for i in range(len(names))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    for ax, mode in zip(axes, ("cycle", "time")):
        for name, col in zip(names, cols):
            t, e = curves[name]
            x = np.arange(len(e)) if mode == "cycle" else t
            ax.plot(x, e, "-o", color=col, lw=2, ms=4, label=f"eta = {lrs[name]:.4g}")
        for rn, v in refs.items():
            ax.axhline(v, color="0.45", lw=1, ls="--")
            ax.text(ax.get_xlim()[1], v, f" {rn}", va="bottom", ha="right", fontsize=8, color="0.3")
        ax.set_ylabel("mean online error in cycle")
        ax.set_xlabel("cycle index" if mode == "cycle" else "eta x steps (cycle mid-points)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, title="step size")
    fig.suptitle(title_note, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def load_curves(exp: Path, methods: list[str], seeds: list[int] | None):
    curves, lrs, ns = {}, {}, {}
    for m in methods:
        hashes = sorted({d.name.rsplit("_seed", 1)[0] for d in (exp / m).glob("*_seed*") if (d / "summary.json").exists()})
        if len(hashes) != 1:
            raise SystemExit(f"{m}: expected exactly one config, found {hashes}")
        rows = []
        for d in sorted((exp / m).glob(f"{hashes[0]}_seed*")):
            s = json.loads((d / "summary.json").read_text())
            if seeds is None or int(s["seed"]) in seeds:
                cfg = agg.read_config(d)
                rows.append((s["err_by_cycle"], cfg))
        if not rows:
            raise SystemExit(f"{m}: no runs for seeds {seeds}")
        cfg = rows[0][1]
        lr = float(cfg["tta"]["lr"])
        spc = int(cfg["stream"]["num_corruptions"]) * (10000 // int(cfg["stream"]["batch_size"]))
        n = min(len(r[0]) for r in rows)
        err = np.mean([r[0][:n] for r in rows], axis=0)
        curves[m], lrs[m], ns[m] = (t_axis(lr, spc, n), err), lr, len(rows)
    return curves, lrs, ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--bn-ref", default="bn_adapt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exp = Path(a.exp_dir)
    curves, lrs, ns = load_curves(exp, a.methods, a.seeds)
    refs = {}
    rdir = exp / a.bn_ref
    if rdir.exists():
        vals = []
        for f in rdir.glob("*_seed*/summary.json"):
            s = json.loads(f.read_text())
            if a.seeds is None or int(s["seed"]) in a.seeds:
                vals.append(s["overall_err"])
        if vals:
            refs[a.bn_ref] = float(np.mean(vals))
    grid = np.array([50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 3000], dtype=float)
    tab = interp_table(curves, grid, a.methods[0])
    out = Path(a.out) if a.out else exp / "dose_response"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "dose_response_table.csv", index=False)
    note = f"seeds used per method: {ns}; reference line = {a.bn_ref} overall error (dashed)"
    plot_curves(curves, lrs, refs, out / "dose_response.png", note)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(tab.round(3).to_string(index=False))
    print(f"\ncycle-0 error by eta: " + ", ".join(f"{lrs[m]:.4g}: {curves[m][1][0]:.3f}" for m in sorted(curves, key=lambda n: lrs[n])))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
