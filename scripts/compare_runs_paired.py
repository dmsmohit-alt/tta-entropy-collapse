"""Paired per-seed comparison of two run groups on the SAME streams (e.g. TENT in nchw vs as_is).

Usage (same method, two configs, e.g. layouts):
    python scripts/compare_runs_paired.py --exp-dir results/layout_study --method tent --hash-a d1ff6da8 --hash-b 787bf53b
Usage (two methods, one cycle; hashes are found automatically when a method has a single config; B - A is printed):
    python scripts/compare_runs_paired.py --exp-dir results/stream_tent --method-a bn_adapt --method-b tent_im_lam1 --cycle 0
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


def list_hashes(exp: Path, method: str) -> list[str]:
    return sorted({d.name.rsplit("_seed", 1)[0] for d in (exp / method).glob("*_seed*") if (d / "summary.json").exists()})


def pick_hash(exp: Path, method: str, h: str | None) -> str:
    if h:
        return h
    hs = list_hashes(exp, method)
    if len(hs) != 1:
        raise SystemExit(f"{method}: {len(hs)} configs found {hs}; pass --hash-a / --hash-b explicitly")
    return hs[0]


def load_group(exp: Path, method: str, h: str) -> dict[int, dict]:
    out = {}
    for f in sorted((exp / method).glob(f"{h}*_seed*/summary.json")):
        s = json.loads(f.read_text())
        out[int(s["seed"])] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--method", default=None, help="same method for A and B (use --method-a/--method-b otherwise)")
    ap.add_argument("--method-a", default=None)
    ap.add_argument("--method-b", default=None)
    ap.add_argument("--hash-a", default=None)
    ap.add_argument("--hash-b", default=None)
    ap.add_argument("--cycle", type=int, default=None, help="compare err_by_cycle[CYCLE] instead of overall / last-cycle error")
    a = ap.parse_args()
    exp = Path(a.exp_dir)
    ma, mb = a.method_a or a.method, a.method_b or a.method
    if not ma or not mb:
        raise SystemExit("give --method, or both --method-a and --method-b")
    ha, hb = pick_hash(exp, ma, a.hash_a), pick_hash(exp, mb, a.hash_b)
    ga, gb = load_group(exp, ma, ha), load_group(exp, mb, hb)
    a.hash_a, a.hash_b = f"{ma}:{ha}", f"{mb}:{hb}"
    seeds = sorted(set(ga) & set(gb))
    if not seeds:
        raise SystemExit("no common seeds between the two groups")
    print(f"A = {a.hash_a} ({len(ga)} runs), B = {a.hash_b} ({len(gb)} runs), common seeds {seeds}")
    keys = [f"err_by_cycle[{a.cycle}]"] if a.cycle is not None else KEYS
    for k in keys:
        get = (lambda g, s: g[s]["err_by_cycle"][a.cycle]) if a.cycle is not None else (lambda g, s, k=k: g[s][k])
        va = np.array([get(ga, s) for s in seeds], dtype=float)
        vb = np.array([get(gb, s) for s in seeds], dtype=float)
        d = vb - va
        print(f"\n{k}:")
        for s, x, y, dd in zip(seeds, va, vb, d):
            print(f"  seed {s}: A={x:.4f}  B={y:.4f}  B-A={dd:+.4f}")
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        print(f"  seeds with B < A: {int((d < 0).sum())} of {len(d)}")
        print(f"  mean(B-A) = {d.mean():+.4f}, std of differences (ddof=1) = {sd:.4f}, n = {len(d)}; "
              f"seed std within A = {va.std(ddof=1) if len(va) > 1 else float('nan'):.4f}, within B = "
              f"{vb.std(ddof=1) if len(vb) > 1 else float('nan'):.4f}")


if __name__ == "__main__":
    main()
