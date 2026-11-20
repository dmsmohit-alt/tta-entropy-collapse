"""Error as a function of the number of steps since the last reset, relative to BN-adapt on the SAME batches.

For a periodic-reset run (reset_every = T, reset_offset = o), batch t is predicted by a model that has had s adaptation steps since the last reset:
    s = t                      if t < T - o      (no reset yet)
    s = (t + o) mod T          otherwise          (s = 0: the first batch after a reset, predicted by the source parameters)
diff(t) = online_err(method, t) - online_err(BN-adapt, t).  Same seed => identical batches, so diff removes the difficulty of the corruption
that happens to sit at step t. At s = 0 the method IS BN-adapt (batch statistics, source affine parameters, prediction before the update),
so diff(s=0) must be 0 up to floating point: this script prints max |diff(s=0)| as a check of the reset wrapper.
Reading: where the mean diff turns positive and stays positive is the horizon after which adapting longer hurts (relative to not adapting at all);
a reset period near that horizon is where a sawtooth should do best. This is descriptive: one stream order per seed, 15-corruption order fixed.
CAVEAT (alignment): every T in our grid is a multiple of 50 = batches per corruption, so resets happen right before a new corruption starts. A reset
at a known domain boundary may help for that reason alone; reset_offset (e.g. 25) tests it.
Usage:
  python scripts/error_vs_steps_since_reset.py --exp-dir results/select_sev3 --methods rdumb_T250 rdumb_T500 rdumb_T1000 rdumb_T2000 --ref bn_adapt --bin 25
Writes error_vs_steps_since_reset.csv and .png into --exp-dir.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_runs as agg  # noqa: E402


def steps_since_reset(step: np.ndarray, period: int, offset: int) -> np.ndarray:
    first_reset = period - offset
    return np.where(step < first_reset, step, (step + offset) % period)


def batch_rows(m: pd.DataFrame, ref: pd.DataFrame, period: int, offset: int, width: int) -> pd.DataFrame:
    cols = ["step", "online_err"] + (["corruption"] if "corruption" in m.columns else [])
    df = m[cols].merge(ref[["step", "online_err"]], on="step", suffixes=("", "_ref"))
    df["s"] = steps_since_reset(df["step"].to_numpy(), period, offset)
    df["diff"] = df["online_err"] - df["online_err_ref"]
    df["bin"] = (df["s"] // width) * width
    return df


def per_run_table(m: pd.DataFrame, ref: pd.DataFrame, period: int, offset: int, width: int) -> pd.DataFrame:
    df = batch_rows(m, ref, period, offset, width)
    g = df.groupby("bin").agg(n=("diff", "size"), err=("online_err", "mean"), err_ref=("online_err_ref", "mean"), diff=("diff", "mean"))
    return g.reset_index().assign(max_abs_diff_s0=float(df.loc[df["s"] == 0, "diff"].abs().max()) if (df["s"] == 0).any() else np.nan)


def n_blocks(corruption_idx: np.ndarray, bins: np.ndarray) -> int:
    """Connected components of the bipartite graph corruption -- s-bin. 1 = an additive 'corruption + s-bin' model is identifiable;
    > 1 = s-bins in different blocks cannot be compared after adjusting for corruption identity (s and corruption are confounded)."""
    k, b = np.unique(corruption_idx, return_inverse=True)[1], np.unique(bins, return_inverse=True)[1]
    K, B = k.max() + 1, b.max() + 1
    r, c = np.concatenate([k, K + b]), np.concatenate([K + b, k])
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(K + B, K + B))
    return int(connected_components(g, directed=False)[0])


def design_check(period: int, offset: int = 0, seg: int = 50, n_corr: int = 15, cycles: int = 10, width: int = 25) -> tuple[int, int]:
    """(connected blocks, number of s-bins) for a periodic-reset schedule on the repeated 15-corruption stream; needs no run data."""
    t = np.arange(seg * n_corr * cycles)
    s = steps_since_reset(t, period, offset)
    return n_blocks((t % (seg * n_corr)) // seg, s // width), int(s.max() // width + 1)


def decompose(rows: pd.DataFrame, width: int) -> pd.DataFrame | None:
    """Additive model diff = alpha_corruption + beta_sbin (least squares, pooled over seeds). None if not identifiable.
    Returns the corruption-adjusted s-profile alpha_mean + beta_bin. Assumes NO corruption x s interaction and no drift across cycles."""
    cor = rows["corruption"].to_numpy()
    if n_blocks(pd.factorize(cor)[0], rows["bin"].to_numpy()) != 1:
        return None
    ci, bi = pd.factorize(cor)[0], pd.factorize(rows["bin"].to_numpy(), sort=True)[0]
    K, B = ci.max() + 1, bi.max() + 1
    X = np.zeros((len(rows), K + B - 1))
    X[np.arange(len(rows)), ci] = 1.0
    nz = bi > 0
    X[np.flatnonzero(nz), K + bi[nz] - 1] = 1.0
    coef = np.linalg.lstsq(X, rows["diff"].to_numpy(), rcond=None)[0]
    alpha_mean = coef[:K].mean()
    beta = np.concatenate([[0.0], coef[K:]])
    bins = np.sort(rows["bin"].unique())
    return pd.DataFrame({"bin_start": bins, "adjusted_diff": alpha_mean + beta})


def lasting_break_even(bins: np.ndarray, diff: np.ndarray) -> int | None:
    """First bin from which the mean diff is > 0 for ALL later bins (None if the last bin is not > 0)."""
    pos = diff > 0
    if not pos[-1]:
        return None
    k = len(pos) - 1
    while k > 0 and pos[k - 1]:
        k -= 1
    return int(bins[k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--design-check", type=int, nargs="+", default=None, metavar="T",
                    help="no data needed: for each T (with --offset) say whether the s-profile can be separated from corruption identity")
    ap.add_argument("--offset", type=int, default=0, help="offset for --design-check")
    ap.add_argument("--decompose", action="store_true", help="also fit diff = corruption + s-bin and print the adjusted profile when identifiable")
    ap.add_argument("--ref", default="bn_adapt")
    ap.add_argument("--bin", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    if a.design_check:
        for T in a.design_check:
            nb_, nbins = design_check(T, a.offset, width=a.bin)
            print(f"T={T} offset={a.offset}: {nbins} s-bins of {a.bin}, blocks = {nb_} -> "
                  f"{'identifiable' if nb_ == 1 else 'NOT separable from corruption identity (5-block structure: the profile mixes s with corruption)'}")
        return
    if not a.exp_dir or not a.methods:
        raise SystemExit("give --exp-dir and --methods (or use --design-check)")
    exp = Path(a.exp_dir)
    rows = []
    pooled: dict[str, list[pd.DataFrame]] = {}
    for method in a.methods:
        per_seed = []
        for d in sorted((exp / method).glob("*_seed*")):
            if not (d / "metrics.csv").exists():
                continue
            seed = int(re.search(r"_seed(\d+)$", d.name).group(1))
            if a.seeds is not None and seed not in a.seeds:
                continue
            tta = agg.read_config(d).get("tta") or {}
            period, offset = tta.get("reset_every"), int(tta.get("reset_offset", 0))
            if not period:
                raise SystemExit(f"{d}: no tta.reset_every in config.yaml (not a periodic-reset run)")
            refs = sorted((exp / a.ref).glob(f"*_seed{seed}"))
            refs = [r for r in refs if (r / "metrics.csv").exists()]
            if len(refs) != 1:
                raise SystemExit(f"reference {a.ref} for seed {seed}: expected 1 run, found {[r.name for r in refs]}")
            mdf, rdf = pd.read_csv(d / "metrics.csv"), pd.read_csv(refs[0] / "metrics.csv")
            if a.decompose:
                pooled.setdefault(method, []).append(batch_rows(mdf, rdf, int(period), offset, a.bin))
            t = per_run_table(mdf, rdf, int(period), offset, a.bin)
            t["seed"] = seed
            per_seed.append(t.assign(T=int(period), offset=offset))
        if not per_seed:
            print(f"{method}: no runs found")
            continue
        allr = pd.concat(per_seed)
        piv = allr.pivot_table(index="bin", columns="seed", values="diff")
        out = pd.DataFrame({"method": method, "T": allr["T"].iloc[0], "offset": allr["offset"].iloc[0], "bin_start": piv.index,
                            "n_runs": piv.shape[1], "mean_err": allr.groupby("bin")["err"].mean().to_numpy(),
                            "mean_err_ref": allr.groupby("bin")["err_ref"].mean().to_numpy(), "mean_diff": piv.mean(axis=1).to_numpy(),
                            "std_diff_over_seeds": (piv.std(axis=1, ddof=1).to_numpy() if piv.shape[1] > 1 else np.nan)})
        rows.append(out)
        run_mean = np.cumsum(out["mean_diff"].to_numpy()) / np.arange(1, len(out) + 1)       # equal-size bins: mean diff of a window ending at this bin
        k = int(np.argmin(run_mean))
        print(f"{method}: running-mean minimum {run_mean[k]:+.4f} at s <= {int(out['bin_start'].iloc[k]) + a.bin - 1} "
              f"(a reset period near there would minimise the mean excess IF the profile transferred; it is confounded with corruption identity, see --decompose)")
        if a.decompose:
            dec = decompose(pd.concat(pooled[method]), a.bin)
            if dec is None:
                print(f"{method}: --decompose: NOT identifiable (blocks > 1): s-profile cannot be separated from corruption identity in this design")
            else:
                print(f"{method}: corruption-adjusted s-profile (additive model, noisy: weakly connected design):")
                print(dec.round(4).to_string(index=False))
        print(f"{method}: T={out['T'].iloc[0]} offset={out['offset'].iloc[0]} runs={piv.shape[1]}  max|diff at s=0| = "
              f"{allr['max_abs_diff_s0'].max():.2e} (must be ~0)  lasting break-even bin: "
              f"{lasting_break_even(out['bin_start'].to_numpy(), out['mean_diff'].to_numpy())}  mean diff over all batches: "
              f"{float(np.average(out['mean_diff'], weights=allr[allr['seed'] == allr['seed'].iloc[0]].groupby('bin')['n'].sum().to_numpy())):+.4f}")
    if not rows:
        raise SystemExit("nothing to analyse")
    res = pd.concat(rows)
    res.to_csv(exp / "error_vs_steps_since_reset.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(res.round(4).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ms = list(dict.fromkeys(res["method"]))
    for i, m in enumerate(ms):
        r = res[res["method"] == m]
        ax.plot(r["bin_start"], r["mean_diff"], "-o", ms=3, lw=2, color=cm.Blues(0.35 + 0.6 * i / max(1, len(ms) - 1)), label=f"{m} (T={r['T'].iloc[0]})")
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("steps since last reset (bin start)")
    ax.set_ylabel(f"error minus {a.ref} (same batches)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(exp / "error_vs_steps_since_reset.png")
    print(f"\nwritten: {exp / 'error_vs_steps_since_reset.csv'} and .png")


if __name__ == "__main__":
    main()
