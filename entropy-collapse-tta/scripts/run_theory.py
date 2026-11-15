"""Phase 1 sweep: map initial bias vs. collapse in the toy model. Writes CSV + summary only.

Usage:
    python scripts/run_theory.py --config configs/toy_basin.yaml [key=value ...]
Plots are made separately:  python scripts/make_plots.py --run-dir <printed run dir>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.theory.toy_model import (
    GDConfig, MixtureSpec, collapse_threshold, is_collapsed, make_mixture, run_with_bias,
)
from src.utils.config import load_config, make_run_dir
from src.utils.run_logger import RunLogger, get_text_logger
from src.utils.seeding import get_rng, seed_everything


def final_row(tr, spec: MixtureSpec, tau: float) -> dict:
    return {
        "H_bar_init": float(tr.H_bar[0]), "H_bar_final": float(tr.H_bar[-1]),
        "top_frac_final": float(tr.top_frac[-1]), "acc_final": float(tr.acc[-1]),
        "b_gap_final": float(tr.b_gap[-1]), "converged": bool(tr.converged),
        "collapsed": is_collapsed(float(tr.H_bar[-1]), spec.K, tau),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_basin.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg)
    log = get_text_logger("theory", run_dir / "run.log")
    t, sw, seeds = cfg["toy"], cfg["sweep"], cfg["experiment"]["seeds"]
    tau = t["collapse_tau"]
    base_spec = MixtureSpec(K=t["K"], d=t["d"], n_per_class=t["n_per_class"], sep=t["sep"], sigma=t["sigma"])
    gd = lambda adapt="b", **kw: GDConfig(  # noqa: E731
        eta=kw.get("eta", t["eta"]), steps=kw.get("steps", t["steps"]), adapt=adapt,
        record_every=t["record_every"], grad_tol=t["grad_tol"])
    log.info("run dir: %s | seeds: %s", run_dir, seeds)

    # --- A) final-state sweeps over initial bias (1D curves for both adapt modes, 2D over sep)
    rows: list[dict] = []
    for sep in sw["sep_grid"]:
        spec = replace(base_spec, sep=float(sep))
        margin = spec.sep**2 / spec.sigma**2
        deltas = np.linspace(0.0, sw["delta_rel_max"] * margin, sw["n_delta"])
        modes = sw["adapt_modes"] if np.isclose(sep, base_spec.sep) else ["b"]
        for seed in seeds:
            seed_everything(seed)
            mix = make_mixture(spec, get_rng(seed))
            for adapt in modes:
                for delta in deltas:
                    tr = run_with_bias(mix, spec, float(delta), gd(adapt))
                    rows.append({"sep": sep, "margin": margin, "adapt": adapt, "seed": seed,
                                 "delta": float(delta), "delta_rel": float(delta / margin),
                                 **final_row(tr, spec, tau)})
        log.info("sweep done: sep=%s", sep)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(run_dir / "basin_sweep.csv", index=False)

    # --- B) threshold delta* per (sep, seed) by grid scan + bisection; plus robustness budgets
    budgets = {"default": {}, "steps_x4": {"steps": 4 * t["steps"]},
               "eta_half_steps_x2": {"eta": t["eta"] / 2, "steps": 2 * t["steps"]}}
    thr_rows: list[dict] = []
    for sep in sw["sep_grid"]:
        spec = replace(base_spec, sep=float(sep))
        margin = spec.sep**2 / spec.sigma**2
        for seed in seeds:
            mix = make_mixture(spec, get_rng(seed))
            for name, kw in budgets.items():
                if name != "default" and not np.isclose(sep, base_spec.sep):
                    continue  # robustness checked at the default separation only
                r = collapse_threshold(mix, spec, gd("b", **kw), sw["delta_rel_max"] * margin,
                                       n_grid=sw["thr_n_grid"], tau=tau, bisect_iters=sw["thr_bisect_iters"])
                thr_rows.append({"sep": sep, "margin": margin, "seed": seed, "budget": name, **r})
        log.info("thresholds done: sep=%s", sep)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(run_dir / "thresholds.csv", index=False)

    # --- C) full trajectories at multiples of delta* (seed = seeds[0], adapt = b, sep = toy.sep)
    seed0 = seeds[0]
    mix = make_mixture(base_spec, get_rng(seed0))
    dstar = float(thr[(thr.sep == base_spec.sep) & (thr.seed == seed0) & (thr.budget == "default")].delta_star.iloc[0])
    traj_rows: list[pd.DataFrame] = []
    if np.isfinite(dstar):
        for ratio in sw["trajectory_ratios"]:
            tr = run_with_bias(mix, base_spec, ratio * dstar, gd("b"))
            traj_rows.append(pd.DataFrame({
                "ratio": ratio, "delta": ratio * dstar, "step": tr.step, "H_bar": tr.H_bar,
                "mean_H": tr.mean_H, "top_frac": tr.top_frac, "acc": tr.acc, "b_gap": tr.b_gap}))
        pd.concat(traj_rows).to_csv(run_dir / "trajectories.csv", index=False)
    else:
        log.warning("no finite delta* at seed %s: trajectories skipped", seed0)

    # --- summary (every number here is computed from the CSVs above)
    ok = thr[(thr.budget == "default") & (thr.status == "ok")]
    summary = {
        "n_runs_sweep": int(len(sweep)),
        "frac_converged_noncollapsed": float(sweep[~sweep.collapsed].converged.mean()) if (~sweep.collapsed).any() else None,
        "frac_converged_collapsed": float(sweep[sweep.collapsed].converged.mean()) if sweep.collapsed.any() else None,
        "max_H_bar_final_among_collapsed": float(sweep[sweep.collapsed].H_bar_final.max()) if sweep.collapsed.any() else None,
        "min_top_frac_among_collapsed": float(sweep[sweep.collapsed].top_frac_final.min()) if sweep.collapsed.any() else None,
        "threshold_by_sep": {str(s): {"mean": float(g.delta_star.mean()), "std": float(g.delta_star.std(ddof=0)),
                                      "ratio_to_margin_mean": float((g.delta_star / g.margin).mean()),
                                      "max_n_switches": int(g.n_switches.max())}
                             for s, g in ok.groupby("sep")},
        "robustness_at_default_sep": {b: {"mean": float(g.delta_star.mean()), "std": float(g.delta_star.std(ddof=0))}
                                       for b, g in thr[np.isclose(thr.sep, base_spec.sep) & (thr.status == "ok")].groupby("budget")},
        "status_counts": thr.status.value_counts().to_dict(),
    }
    with RunLogger(run_dir) as rl:
        rl.write_summary(summary)
    log.info("summary written to %s", run_dir / "summary.json")
    print(run_dir)


if __name__ == "__main__":
    main()
