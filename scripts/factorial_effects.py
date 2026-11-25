"""Phase 5 factorial: paired per-seed effect of the marginal term at each reset period, and their difference (interaction).

effect(T) = err(lam=0, T) - err(lam=1, T) per seed (same batches), then mean over seeds; interaction = effect(1000) - effect(250).
Uses summary.json of results/stream_tent (overall_err). Prints per-seed values; no p-value (n = 3).
Usage: python scripts/factorial_effects.py --exp-dir results/stream_tent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CELLS = {1000: ("rdumb_T1000", "tent_im_lam1_T1000"), 250: ("rdumb_T250", "tent_im_lam1_T250")}


def overall(exp: Path, method: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for f in sorted((exp / method).glob("*_seed*/summary.json")):
        s = json.loads(f.read_text())
        out[int(s["seed"])] = float(s["overall_err"])
    if not out:
        raise SystemExit(f"no runs found for {method} in {exp}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="results/stream_tent")
    a = ap.parse_args()
    exp = Path(a.exp_dir)
    eff: dict[int, dict[int, float]] = {}
    for T, (m0, m1) in CELLS.items():
        e0, e1 = overall(exp, m0), overall(exp, m1)
        seeds = sorted(set(e0) & set(e1))
        eff[T] = {s: e0[s] - e1[s] for s in seeds}
        print(f"T={T}: effect = err({m0}) - err({m1}) (positive = marginal term helps)")
        for s in seeds:
            print(f"  seed {s}: lam0={e0[s]:.4f} lam1={e1[s]:.4f} effect={eff[T][s]:+.4f}")
        v = np.array(list(eff[T].values()))
        print(f"  mean effect {v.mean():+.4f}  std(ddof=1) {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}  n={len(v)}")
    seeds = sorted(set(eff[1000]) & set(eff[250]))
    d = np.array([eff[1000][s] - eff[250][s] for s in seeds])
    print("\ninteraction = effect(1000) - effect(250), per seed: " + ", ".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, d)))
    print(f"mean {d.mean():+.4f} (pre-registered P3: >= 0.05), std(ddof=1) {d.std(ddof=1) if len(d) > 1 else float('nan'):.4f}, n={len(d)}")


if __name__ == "__main__":
    main()
