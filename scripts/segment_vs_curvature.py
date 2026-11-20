"""Does the extra degradation at a larger step size concentrate in segments whose measured local eta* is below that step size?

Idea: the theory's eta* = 2/lambda_max is LOCAL (per segment: 0.29 for the sharpest, ~0.8 for the flattest, measured at the source weights).
For a run with step eta_hi vs a reference run with eta_lo (same seed, same batches), per-corruption error excess = err_hi - err_lo (cycle 0 only).
If eta* matters, the excess should be larger in segments with 2/lambda_max < eta_hi than in the rest, and rank-correlate with lambda_max.
Usage:
  python scripts/segment_vs_curvature.py --exp-dir results/stream_tent --lo tent_im_sgd_lam1 --hi tent_im_sgd_lam1_c2p0 \
      --curvature results/curvature/<name>/curvature.csv --seed 0 --lam 1.0
LIMITS (printed again at the end): 15 points; cycle 0 only, but later segments inherit drift from earlier ones (ORDER confound: the correlation with
the segment index is printed too); curvature measured on a severity-3 batch at source weights, the stream is severity 5; one seed; Spearman, no p-value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cycle0_error(metrics_csv: Path) -> pd.Series:
    df = pd.read_csv(metrics_csv)
    df = df[df["cycle"] == 0]
    return df.groupby("corruption", sort=False)["online_err"].mean()


def analyse(err_lo: pd.Series, err_hi: pd.Series, curv: pd.DataFrame, eta_hi: float) -> dict:
    c = curv.set_index("corruption")
    idx = [k for k in err_lo.index if k in c.index]
    ex = (err_hi[idx] - err_lo[idx])
    lam = c.loc[idx, "lam_max"].astype(float)
    order = pd.Series(range(len(idx)), index=idx)
    above = (2.0 / lam) < eta_hi                                  # segments where eta_hi exceeds the local GD limit
    rho_lam = spearmanr(lam, ex)[0] if len(idx) > 2 else float("nan")
    rho_ord = spearmanr(order, ex)[0] if len(idx) > 2 else float("nan")
    return {"table": pd.DataFrame({"excess": ex, "lam_max": lam, "eta_star_local": 2.0 / lam, "above_eta_star": above}),
            "n": len(idx), "n_above": int(above.sum()), "excess_above": float(ex[above].mean()) if above.any() else float("nan"),
            "excess_below": float(ex[~above].mean()) if (~above).any() else float("nan"), "rho_lam_max": float(rho_lam),
            "rho_segment_index": float(rho_ord)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--lo", required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--curvature", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    exp = Path(a.exp_dir)

    def run_dir(m: str) -> Path:
        ds = sorted(d for d in (exp / m).glob(f"*_seed{a.seed}") if (d / "metrics.csv").exists())
        if len(ds) != 1:
            raise SystemExit(f"{m}: expected one run for seed {a.seed}, found {[d.name for d in ds]}")
        return ds[0]

    import yaml
    d_hi = run_dir(a.hi)
    eta_hi = float(yaml.safe_load((d_hi / "config.yaml").read_text())["tta"]["lr"])
    curv = pd.read_csv(a.curvature)
    curv = curv[np.isclose(curv["lam"], a.lam)]
    r = analyse(cycle0_error(run_dir(a.lo) / "metrics.csv"), cycle0_error(d_hi / "metrics.csv"), curv, eta_hi)
    with pd.option_context("display.width", 200):
        print(r["table"].round(4).to_string())
    print(f"\neta_hi = {eta_hi}; segments above their local eta*: {r['n_above']} of {r['n']}")
    print(f"mean excess error (hi - lo): above local eta* = {r['excess_above']:.4f}, below = {r['excess_below']:.4f}")
    print(f"Spearman(excess, lambda_max) = {r['rho_lam_max']:.2f};  Spearman(excess, segment index) = {r['rho_segment_index']:.2f}  <- order confound")
    print("Limits: 15 points, cycle 0, later segments inherit drift, curvature from severity 3 at source weights, one seed, no p-value.")


if __name__ == "__main__":
    main()
