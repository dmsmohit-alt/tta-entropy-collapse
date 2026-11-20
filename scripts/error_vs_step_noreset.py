"""Error minus BN-adapt (same batches) against the global step, for methods WITHOUT resets (tent, tent_im_lam1, tent_im_sgd_lam1, ...).

Answers: how long does adapting help before it starts to hurt, relative to not adapting at all (BN-adapt)? Same seed => identical batches, so
diff(t) = err_method(t) - err_bn_adapt(t) removes the difficulty of the corruption at step t. Resolution is --bin steps (default 25 = 5000 samples per seed per bin).
Because the stream repeats the 15 corruptions every cycle, the global step is NOT confounded with corruption identity across cycles (cycle x corruption grid),
but WITHIN the first cycle step and corruption identity are the same thing: a dip in cycle 0 can be a corruption effect. The cycle x corruption table is written for that.
Break-even is reported on a TRAILING rolling mean (--roll bins, default 6 = 150 steps), because single 25-step bins are noisy:
 'first positive' = first step whose trailing mean excess is > 0; 'lasting' = from there on it is > 0 for every later bin in the range examined.
Usage:
  python scripts/error_vs_step_noreset.py --exp-dir results/stream_tent --methods tent tent_im_lam1 tent_im_sgd_lam1 --ref bn_adapt --bin 25 --max-step 3000
Writes error_vs_step_noreset.csv (per bin), error_vs_step_noreset_grid.csv (cycle x corruption) and error_vs_step_noreset.png into --exp-dir.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def break_even(end_steps: np.ndarray, roll: np.ndarray) -> tuple[int | None, int | None]:
    """(first step with trailing-mean > 0, first step from which it stays > 0 until the end of the range). None if never / not lasting."""
    ok = ~np.isnan(roll)
    pos = (roll > 0) & ok
    first = int(end_steps[np.flatnonzero(pos)[0]]) if pos.any() else None
    if not pos.any() or not pos[ok][-1]:
        return first, None
    idx = np.flatnonzero(ok)
    k = len(idx) - 1
    while k > 0 and pos[idx[k - 1]]:
        k -= 1
    return first, int(end_steps[idx[k]])


def diffs_for_run(m: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    df = m[["step", "cycle", "corruption", "online_err"]].merge(ref[["step", "online_err"]], on="step", suffixes=("", "_ref"))
    df["diff"] = df["online_err"] - df["online_err_ref"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ref", default="bn_adapt")
    ap.add_argument("--bin", type=int, default=25)
    ap.add_argument("--roll", type=int, default=6)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    exp = Path(a.exp_dir)
    bins_out, grid_out = [], []
    for method in a.methods:
        per_seed = []
        for d in sorted((exp / method).glob("*_seed*")):
            if not (d / "metrics.csv").exists():
                continue
            seed = int(re.search(r"_seed(\d+)$", d.name).group(1))
            if a.seeds is not None and seed not in a.seeds:
                continue
            refs = [r for r in (exp / a.ref).glob(f"*_seed{seed}") if (r / "metrics.csv").exists()]
            if len(refs) != 1:
                raise SystemExit(f"reference {a.ref} for seed {seed}: expected 1 run, found {[r.name for r in refs]}")
            df = diffs_for_run(pd.read_csv(d / "metrics.csv"), pd.read_csv(refs[0] / "metrics.csv"))
            if a.max_step is not None:
                df = df[df["step"] < a.max_step]
            df["bin"] = (df["step"] // a.bin) * a.bin
            df["seed"] = seed
            per_seed.append(df)
        if not per_seed:
            print(f"{method}: no runs found")
            continue
        allr = pd.concat(per_seed)
        piv = allr.groupby(["bin", "seed"])["diff"].mean().unstack("seed")
        mean = piv.mean(axis=1).to_numpy()
        roll = trailing_mean(mean, a.roll)
        end_steps = piv.index.to_numpy() + a.bin - 1
        first, lasting = break_even(end_steps, roll)
        print(f"{method}: seeds={piv.shape[1]}, steps examined 0..{int(end_steps[-1])}; trailing {a.roll * a.bin}-step mean excess vs {a.ref}: "
              f"first > 0 at step {first}, lasting > 0 from step {lasting}; minimum {np.nanmin(roll):+.4f} at step {int(end_steps[int(np.nanargmin(roll))])}")
        cyc = allr.groupby(["cycle", "seed"])["diff"].mean().unstack("seed").mean(axis=1)
        print("   mean excess by cycle: " + ", ".join(f"c{int(c)}:{v:+.4f}" for c, v in cyc.items()))
        bins_out.append(pd.DataFrame({"method": method, "bin_start": piv.index, "n_runs": piv.shape[1], "mean_diff": mean,
                                      "std_over_seeds": piv.std(axis=1, ddof=1).to_numpy() if piv.shape[1] > 1 else np.nan, "roll_diff": roll}))
        g = allr.groupby(["cycle", "corruption"], sort=False)["diff"].mean().reset_index()
        grid_out.append(g.assign(method=method))
    if not bins_out:
        raise SystemExit("nothing to analyse")
    res = pd.concat(bins_out)
    res.to_csv(exp / "error_vs_step_noreset.csv", index=False)
    pd.concat(grid_out).to_csv(exp / "error_vs_step_noreset_grid.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    cols = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
    for i, (m, r) in enumerate(res.groupby("method", sort=False)):
        ax.plot(r["bin_start"] + a.bin / 2, r["roll_diff"], lw=2, color=cols[i % len(cols)], label=m)
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("global step (bin centre)")
    ax.set_ylabel(f"error minus {a.ref}, trailing {a.roll * a.bin}-step mean")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(exp / "error_vs_step_noreset.png")
    print(f"\nwritten: {exp / 'error_vs_step_noreset.csv'}, ..._grid.csv, .png")


if __name__ == "__main__":
    main()
