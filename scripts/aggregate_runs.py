"""Aggregate per-seed summaries into mean +/- std tables (no recomputation: reads summary.json only).

Usage:  python scripts/aggregate_runs.py --exp-dir results/stream_tent
Groups runs by (method, config hash without seed). std is the SAMPLE std (ddof=1); with fewer than 2 runs it is
reported as NaN. Collapse time is summarised over runs that collapsed only, together with n_collapsed / n_runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KEYS = ["overall_err", "err_last_cycle", "final_acc_last_cycle", "max_top_frac", "mean_top_frac_last_cycle",
        "final_H_bar_ema", "seconds"]


def layout_of(run_dir: Path, summary: dict) -> str:
    """Layout a run used: summary.json first, then config.yaml (compute.input_layout); 'unrecorded' if neither has it.
    If both exist and disagree we say so instead of picking one."""
    from_summary = summary.get("input_layout")
    from_config = None
    cfg_file = run_dir / "config.yaml"
    if cfg_file.exists():
        try:
            from_config = (yaml.safe_load(cfg_file.read_text()) or {}).get("compute", {}).get("input_layout")
        except yaml.YAMLError:
            pass
    if from_summary and from_config and from_summary != from_config:
        return f"CONFLICT(summary={from_summary},config={from_config})"
    return from_summary or from_config or "unrecorded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    exp = Path(ap.parse_args().exp_dir)
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(exp.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        s["_layout"] = layout_of(f.parent, s)
        groups.setdefault((f.parent.parent.name, f.parent.name.rsplit("_seed", 1)[0]), []).append(s)
    rows = []
    for (method, h), runs in groups.items():
        row: dict = {"method": method, "layout": "/".join(sorted({r["_layout"] for r in runs})), "config_hash": h, "n_runs": len(runs),
                     "seeds": sorted(r["seed"] for r in runs)}
        for k in KEYS:
            v = np.array([r[k] for r in runs if r.get(k) is not None], dtype=float)
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        tc = np.array([r["time_to_collapse_start"] for r in runs if r.get("time_to_collapse_start") is not None], dtype=float)
        row.update({"n_collapsed": len(tc), "ttc_start_mean": float(tc.mean()) if len(tc) else np.nan,
                    "ttc_start_std": float(tc.std(ddof=1)) if len(tc) > 1 else np.nan,
                    "n_diverged": sum(r.get("diverged_step") is not None for r in runs)})
        rows.append(row)
    df = pd.DataFrame(rows)
    out = exp / "aggregate.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df.round(4).to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
