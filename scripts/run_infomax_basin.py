"""Toy check: GD on  E[H] - lam*H(p_bar)  from the Bayes classifier + bias delta on class 0 (bias-only).

Usage:  python scripts/run_infomax_basin.py --config configs/toy_infomax_basin.yaml [key=value ...]
Writes infomax_basin.csv and summary.json. TOY MODEL ONLY: says nothing yet about deep networks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.theory.stability import SampleProblem
from src.theory.toy_model import MixtureSpec, bayes_init, entropy_stats, is_collapsed, make_mixture
from src.utils.config import load_config, make_run_dir
from src.utils.run_logger import RunLogger, get_text_logger
from src.utils.seeding import get_rng, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_infomax_basin.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg)
    log = get_text_logger("infomax", run_dir / "run.log")
    t, c, seeds = cfg["toy"], cfg["check"], cfg["experiment"]["seeds"]
    rows: list[dict] = []
    for sep in c["sep_grid"]:
        spec = MixtureSpec(K=t["K"], d=t["d"], n_per_class=t["n_per_class"], sep=float(sep), sigma=t["sigma"])
        margin = sep**2 / t["sigma"] ** 2
        for seed in seeds:
            seed_everything(seed)
            mix = make_mixture(spec, get_rng(seed))
            w, _ = bayes_init(mix.means, spec.sigma)
            z0 = mix.X @ w.T
            for lam in c["lam_list"]:
                prob = SampleProblem(z0, lam)
                for rel in c["delta_rel"]:
                    b = np.zeros(t["K"]); b[0] = rel * margin
                    for _ in range(c["steps"]):
                        b = b - c["eta"] * prob.grad(b)
                    st = entropy_stats(prob.logp(b), mix.y)
                    rows.append({"sep": sep, "seed": seed, "lam": lam, "delta_rel": rel, "eta": c["eta"],
                                 "steps": c["steps"], "H_bar_final": st["H_bar"], "top_frac_final": st["top_frac"],
                                 "acc_final": st["acc"], "collapsed": is_collapsed(st["H_bar"], t["K"], c["collapse_tau"])})
        log.info("done sep=%s", sep)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "infomax_basin.csv", index=False)
    summary = {"collapse_fraction": {f"sep={k[0]:g},lam={k[1]:g},delta_rel={k[2]:g}": float(v)
                                     for k, v in df.groupby(["sep", "lam", "delta_rel"]).collapsed.mean().items()},
               "mean_acc_final": {f"sep={k[0]:g},lam={k[1]:g},delta_rel={k[2]:g}": float(v)
                                  for k, v in df.groupby(["sep", "lam", "delta_rel"]).acc_final.mean().items()}}
    with RunLogger(run_dir) as rl:
        rl.write_summary(summary)
    print(run_dir)


if __name__ == "__main__":
    main()
