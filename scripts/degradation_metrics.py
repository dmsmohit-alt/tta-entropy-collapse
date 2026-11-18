"""Error-based failure metric: when does a method become worse than a reference, cycle by cycle?

Why: the top-class-fraction rule (src/metrics/collapse.py) detects MARGINAL collapse only. A run can keep a balanced marginal
(H(p_bar) ~ log K, top_frac ~ 1/K) and still be badly wrong (tent_im_sgd_lam1 with eta > eta*: error 0.80, top_frac 0.13).
This script defines that second failure mode, "degradation", from error alone.

DEFINITION (fixed here; written down after seeing the Phase 4d per-cycle means, so it is NOT blind for those runs, but it is
fixed before any later run: c = 0.9 / 1.1 / 0.25, the severity-3 selection and the sweeps):
  - unit = one cycle (all corruptions once; 750 steps at batch 200); err[c] = mean online error in cycle c.
  - reference = BN-adapt (primary) and Source (secondary), same seeds, same stream, same layout (matched through the saved config.yaml:
    data/stream/model/compute, ignoring num_cycles and max_steps; cycles missing in the reference are not judged).
  - excess[c] = mean over common seeds of err_method[c] - err_ref[c].
  - PRIMARY delta = 0.01 (absolute error): degraded in cycle c  <=>  excess[c] > 0.01. Sensitivity columns: delta in {0, 0.02, 0.05}.
    0.01 is far above the BN-adapt seed noise (std ~1e-4) and far below the effects seen (>= 0.03); it was not tuned on a score.
  - degradation_onset = first degraded cycle (None if never); sustained = every cycle from the onset on is degraded;
    n_degraded = number of degraded cycles; final_excess = excess in the last judged cycle.
  - per-seed onsets use the SAME seed's reference run (identical batches), so they are paired.
Usage:  python scripts/degradation_metrics.py --exp-dir results/stream_tent    -> degradation.csv in that folder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_runs as agg  # noqa: E402  (reuses read_config / describe_tta / layout_of)

PRIMARY_DELTA = 0.01
SENSITIVITY = [0.0, 0.01, 0.02, 0.05]
REFS = ["bn_adapt", "source"]


def ctx_key(cfg: dict) -> str:
    """Identity of the stream/model/layout a run saw (num_cycles and max_steps ignored: cycle c is identical for any length)."""
    stream = {k: v for k, v in (cfg.get("stream") or {}).items() if k not in ("num_cycles", "max_steps")}
    return json.dumps({"data": cfg.get("data"), "stream": stream, "model": cfg.get("model"), "compute": cfg.get("compute")},
                      sort_keys=True, default=str)


def load_groups(exp: Path) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict] = {}
    for f in sorted(exp.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        cfg = agg.read_config(f.parent)
        key = (f.parent.parent.name, f.parent.name.rsplit("_seed", 1)[0])
        g = groups.setdefault(key, {"seeds": {}, "ctx": ctx_key(cfg), "tta": agg.describe_tta(cfg), "layout": agg.layout_of(f.parent, s)})
        if s.get("err_by_cycle"):
            g["seeds"][int(s["seed"])] = [float(x) for x in s["err_by_cycle"]]
    return groups


def first_above(excess: np.ndarray, delta: float) -> int | None:
    idx = np.flatnonzero(excess > delta)
    return int(idx[0]) if len(idx) else None


def judge(method_runs: dict[int, list[float]], ref_runs: dict[int, list[float]], delta: float) -> dict | None:
    seeds = sorted(set(method_runs) & set(ref_runs))
    if not seeds:
        return None
    n = min(min(len(method_runs[s]), len(ref_runs[s])) for s in seeds)
    m = np.array([method_runs[s][:n] for s in seeds])
    r = np.array([ref_runs[s][:n] for s in seeds])
    excess = (m - r).mean(0)
    onset = first_above(excess, delta)
    per_seed = [first_above(m[i] - r[i], delta) for i in range(len(seeds))]
    return {"n_seeds": len(seeds), "n_cycles_judged": n, "onset": onset,
            "sustained": None if onset is None else bool((excess[onset:] > delta).all()),
            "n_degraded": int((excess > delta).sum()), "final_excess": float(excess[-1]),
            "per_seed_onset": per_seed, "seeds": seeds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--delta", type=float, default=PRIMARY_DELTA)
    exp = Path(ap.parse_args().exp_dir)
    delta = ap.parse_args().delta
    groups = load_groups(exp)
    rows = []
    for (method, h), g in groups.items():
        if method in REFS:
            continue
        row = {"method": method, "tta": g["tta"], "layout": g["layout"], "config_hash": h}
        for ref in REFS:
            cands = [(k, v) for k, v in groups.items() if k[0] == ref and v["ctx"] == g["ctx"]]
            if not cands:
                row[f"onset_vs_{ref}"] = "no matching reference run"
                continue
            res = judge(g["seeds"], cands[0][1]["seeds"], delta)
            if res is None:
                row[f"onset_vs_{ref}"] = "no common seeds"
                continue
            row.update({f"onset_vs_{ref}": res["onset"], f"sustained_vs_{ref}": res["sustained"],
                        f"n_degraded_vs_{ref}": res["n_degraded"], f"final_excess_vs_{ref}": round(res["final_excess"], 4),
                        f"per_seed_onset_vs_{ref}": res["per_seed_onset"], f"cycles_judged_vs_{ref}": res["n_cycles_judged"],
                        f"n_seeds_vs_{ref}": res["n_seeds"]})
            if ref == "bn_adapt":
                for d in SENSITIVITY:
                    row[f"onset_vs_bn_adapt_delta{d}"] = judge(g["seeds"], cands[0][1]["seeds"], d)["onset"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(exp / "degradation.csv", index=False)
    show = [c for c in df.columns if not c.startswith("per_seed") and "delta" not in c]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f"degradation vs reference (primary delta = {delta}; onset = first cycle with seed-mean excess error > delta; None = never)")
        print(df[show].to_string(index=False))
        print("\nper-seed onsets (paired with the same seed's reference run):")
        print(df[["method", "tta"] + [c for c in df.columns if c.startswith("per_seed")]].to_string(index=False))
    print(f"\nwritten: {exp / 'degradation.csv'}")


if __name__ == "__main__":
    main()
