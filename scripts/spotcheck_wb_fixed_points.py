"""Phase 1 follow-up (README TBD): are the INTERMEDIATE end states of the W+b sweep true fixed points?

For each (delta, seed) of the sep = toy.sep, adapt = "Wb" sweep (same mixture, same grid, same budget as run_theory.py):
  1. run the sweep GD to its budget -> state S (W, b);
  2. gradient norm of the entropy loss at S (over W and b);
  3. CONTINUE from S for `continue_factor` x the budget: change of H(p_bar), fraction of predictions that change, growth of ||W||
     and raw parameter drift. With W adapted the weights can keep growing slowly (logit scale) while the gradient is ~0 and the
     predictions do not change, so raw parameter drift alone does NOT mean 'not a fixed point': we report it but do not use it;
  4. PERTURB S by Gaussian noise of norm `eps`, run the same continuation, and compare the end point with the UNPERTURBED
     continuation's end point (distance < eps => perturbation shrank; > eps => it grew).
A state is called "stationary" if grad norm < grad_loose, |dH_bar| < dh_tol and the prediction change < pred_tol after continuation.
Thresholds are our choice (printed in the summary); "intermediate" := collapse_tau*log K <= H_bar(S) <= 0.9*log K, also our choice.
A growing perturbation shows instability; a shrinking one does NOT prove stability (3 random directions per state).
TOY MODEL ONLY. Usage:
    python scripts/spotcheck_wb_fixed_points.py --config configs/toy_basin.yaml experiment.name=toy_wb_spotcheck
Smoke test: ... toy.n_per_class=50 sweep.n_delta=5 experiment.seeds=[0]
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
    GDConfig, MixtureSpec, bayes_init, entropy_loss_and_grad, entropy_stats, forward_logprobs, make_mixture, run_gd,
)
from src.utils.config import load_config, make_run_dir
from src.utils.run_logger import RunLogger, get_text_logger
from src.utils.seeding import get_rng, seed_everything

CONTINUE_FACTOR = 4      # continuation budget = 4 x the sweep budget
EPS = 1e-2               # norm of the random perturbation (parameters W and b together)
N_PERT = 3
GRAD_LOOSE = 1e-4
DH_TOL = 1e-3
PRED_TOL = 0.005       # fraction of samples whose argmax changes during the continuation


def gnorm(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    _, gW, gb = entropy_loss_and_grad(X, W, b)
    return float(np.sqrt((gW**2).sum() + (gb**2).sum()))


def hbar(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    return entropy_stats(forward_logprobs(X, W, b), y)["H_bar"]


def dist(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> float:
    return float(np.sqrt(((W1 - W2) ** 2).sum() + ((b1 - b2) ** 2).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_basin.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg)
    log = get_text_logger("spotcheck", run_dir / "run.log")
    t, sw, seeds = cfg["toy"], cfg["sweep"], cfg["experiment"]["seeds"]
    spec = MixtureSpec(K=t["K"], d=t["d"], n_per_class=t["n_per_class"], sep=t["sep"], sigma=t["sigma"])
    margin = spec.sep**2 / spec.sigma**2
    deltas = np.linspace(0.0, sw["delta_rel_max"] * margin, sw["n_delta"])   # same grid as run_theory.py
    gd = GDConfig(eta=t["eta"], steps=t["steps"], adapt="Wb", record_every=10**9, grad_tol=t["grad_tol"])
    gd_cont = replace(gd, steps=CONTINUE_FACTOR * t["steps"])
    logK, tau = float(np.log(t["K"])), t["collapse_tau"]
    rows: list[dict] = []
    for seed in seeds:
        seed_everything(seed)
        mix = make_mixture(spec, get_rng(seed))
        rng = np.random.default_rng(1000 + seed)
        for delta in deltas:
            W0, b0 = bayes_init(mix.means, spec.sigma)
            b0 = b0.copy(); b0[0] += float(delta)
            S = run_gd(mix.X, mix.y, W0, b0, gd)
            h0 = hbar(mix.X, mix.y, S.W, S.b)
            g0 = gnorm(mix.X, S.W, S.b)
            C = run_gd(mix.X, mix.y, S.W, S.b, gd_cont)
            h1 = hbar(mix.X, mix.y, C.W, C.b)
            ret = []
            pred0 = forward_logprobs(mix.X, S.W, S.b).argmax(1)
            pred_change = float((forward_logprobs(mix.X, C.W, C.b).argmax(1) != pred0).mean())
            for _ in range(N_PERT):
                nW, nb = rng.normal(size=S.W.shape), rng.normal(size=S.b.shape)
                s = EPS / np.sqrt((nW**2).sum() + (nb**2).sum())
                P = run_gd(mix.X, mix.y, S.W + s * nW, S.b + s * nb, gd_cont)
                ret.append(dist(P.W, P.b, C.W, C.b))
            rows.append({"seed": seed, "delta": float(delta), "delta_rel": float(delta / margin), "H_bar_S": h0, "grad_norm_S": g0,
                         "dH_bar_cont": h1 - h0, "pred_change_cont": pred_change, "W_norm_S": float(np.linalg.norm(S.W)),
                         "W_norm_cont": float(np.linalg.norm(C.W)), "param_drift_cont": dist(C.W, C.b, S.W, S.b),
                         "dist_perturbed_vs_unperturbed_max": float(max(ret)), "eps": EPS,
                         "intermediate": bool(tau * logK <= h0 <= 0.9 * logK),
                         "collapsed": bool(h0 < tau * logK),
                         "stationary": bool(g0 < GRAD_LOOSE and abs(h1 - h0) < DH_TOL and pred_change < PRED_TOL)})
        log.info("seed %s done", seed)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "wb_spotcheck.csv", index=False)
    inter = df[df.intermediate]
    summary = {
        "n_states": int(len(df)), "n_intermediate": int(len(inter)),
        "n_collapsed": int(df.collapsed.sum()),
        "intermediate_stationary": int(inter.stationary.sum()),
        "intermediate_max_grad_norm": float(inter.grad_norm_S.max()) if len(inter) else None,
        "intermediate_max_abs_dH_bar": float(inter.dH_bar_cont.abs().max()) if len(inter) else None,
        "intermediate_max_param_drift": float(inter.param_drift_cont.max()) if len(inter) else None,
        "intermediate_max_dist_perturbed_vs_unperturbed": float(inter.dist_perturbed_vs_unperturbed_max.max()) if len(inter) else None,
        "eps": EPS, "n_perturbations": N_PERT, "continue_factor": CONTINUE_FACTOR, "grad_loose": GRAD_LOOSE, "dh_tol": DH_TOL, "pred_tol": PRED_TOL,
        "intermediate_max_pred_change": float(inter.pred_change_cont.max()) if len(inter) else None,
        "intermediate_max_W_norm_growth": float((inter.W_norm_cont / inter.W_norm_S).max()) if len(inter) else None,
        "intermediate_definition": "tau*logK <= H_bar(S) <= 0.9*logK",
    }
    with RunLogger(run_dir) as rl:
        rl.write_summary(summary)
    pd.set_option("display.width", 200)
    print(inter[["seed", "delta_rel", "H_bar_S", "grad_norm_S", "dH_bar_cont", "pred_change_cont", "W_norm_S", "W_norm_cont", "dist_perturbed_vs_unperturbed_max", "stationary"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print({k: v for k, v in summary.items()})
    print(run_dir)


if __name__ == "__main__":
    main()
