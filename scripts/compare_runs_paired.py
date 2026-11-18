"""Paired per-seed comparison of two run groups on the SAME streams (e.g. TENT in nchw vs as_is).

Usage:
    python scripts/compare_runs_paired.py --exp-dir results/layout_study --method tent --hash-a d1ff6da8 --hash-b 787bf53b
Why paired: for a given seed both groups see the identical batches, so the per-seed DIFFERENCE removes stream-order variance;
comparing the two seed-stds (std of A vs std of B) does not. With n seeds the test has little power; we report the numbers
(per-seed values, mean difference, sample std of differences, n) and no p-value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["overall_err", "err_last_cycle"]


def load_group(exp: Path, method: str, h: str) -> dict[int, dict]:
    out = {}
    for f in sorted((exp / method).glob(f"{h}*_seed*/summary.json")):
        s = json.loads(f.read_text())
        out[int(s["seed"])] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--hash-a", required=True)
    ap.add_argument("--hash-b", required=True)
    a = ap.parse_args()
    exp = Path(a.exp_dir)
    ga, gb = load_group(exp, a.method, a.hash_a), load_group(exp, a.method, a.hash_b)
    seeds = sorted(set(ga) & set(gb))
    if not seeds:
        raise SystemExit("no common seeds between the two groups")
    print(f"A = {a.hash_a} ({len(ga)} runs), B = {a.hash_b} ({len(gb)} runs), common seeds {seeds}")
    for k in KEYS:
        va = np.array([ga[s][k] for s in seeds], dtype=float)
        vb = np.array([gb[s][k] for s in seeds], dtype=float)
        d = vb - va
        print(f"\n{k}:")
        for s, x, y, dd in zip(seeds, va, vb, d):
            print(f"  seed {s}: A={x:.4f}  B={y:.4f}  B-A={dd:+.4f}")
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        print(f"  mean(B-A) = {d.mean():+.4f}, std of differences (ddof=1) = {sd:.4f}, n = {len(d)}; "
              f"seed std within A = {va.std(ddof=1) if len(va) > 1 else float('nan'):.4f}, within B = "
              f"{vb.std(ddof=1) if len(vb) > 1 else float('nan'):.4f}")


if __name__ == "__main__":
    main()
