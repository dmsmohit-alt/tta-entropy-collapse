"""Phase 2: population constants, reduced-flow threshold, balanced-point Hessian, eta*.

Usage:  python scripts/run_stability.py --config configs/toy_stability.yaml [key=value ...]
Writes population.csv, stability.csv, sim_from_zero.csv, sample_roots.csv, summary.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.theory import dynamics as D
from src.theory import stability as S
from src.theory.toy_model import GDConfig, MixtureSpec, bayes_init, entropy_stats, is_collapsed, make_mixture
from src.utils.config import load_config, make_run_dir
from src.utils.run_logger import RunLogger, get_text_logger
from src.utils.seeding import get_rng, seed_everything


def sim_from_zero(prob: S.SampleProblem, eta: float, steps: int) -> dict:
    """Plain GD on J from b = 0 (the balanced point, delta = 0)."""
    b = np.zeros(prob.K)
    for _ in range(steps):
        b = b - eta * prob.grad(b)
    st = entropy_stats(prob.logp(b))
    return {"H_bar_final": st["H_bar"], "top_frac_final": st["top_frac"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_stability.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg)
    log = get_text_logger("stability", run_dir / "run.log")
    t, st, pp = cfg["toy"], cfg["stability"], cfg["population"]
    K, seeds, tau = t["K"], cfg["experiment"]["seeds"], st["collapse_tau"]
    log.info("run dir %s | K=%d seeds=%s", run_dir, K, seeds)

    # ---- A) population constants and reduced-flow basin boundary (Monte Carlo, exact reduction)
    prow: list[dict] = []
    pop_csv = run_dir / "population.csv"
    reuse_pop = pop_csv.exists()          # same config hash => same deterministic result: skip the slow part
    if reuse_pop:
        log.info("reusing existing %s (same config hash; delete it to recompute)", pop_csv)
    for sep in ([] if reuse_pop else st["sep_grid"]):
        m = sep**2 / t["sigma"] ** 2
        for ms in pp["mc_seeds"]:
            pop = D.Population(K, m, n_xi=pp["n_xi"], seed=ms)
            a_l, s = pop.alpha_L(), pop.s_const()
            for lam in st["lam_list"]:
                root = D.critical_gap(pop, lam, pp["g_max_rel"], pp["n_scan"])
                a_j = a_l + lam * K * s**2
                prow.append({"sep": sep, "margin": m, "mc_seed": ms, "lam": lam, "alpha_L": a_l, "s": s,
                             "lam_critical": pop.lam_critical(), "alpha_J": a_j, "eta_star_pop": D.eta_star(a_j),
                             "g_c": root.g_c, "g_c_status": root.status, "n_up_crossings": root.n_up_crossings})
        log.info("population done: sep=%s", sep)
    popdf = pd.read_csv(pop_csv) if reuse_pop else pd.DataFrame(prow)
    popdf.to_csv(pop_csv, index=False)

    # ---- B) sample-level Hessians, eta*, spectral radii; C) GD from b=0
    srow: list[dict] = []
    zrow: list[dict] = []
    probs: dict[tuple, S.SampleProblem] = {}
    for sep in st["sep_grid"]:
        spec = MixtureSpec(K=K, d=t["d"], n_per_class=t["n_per_class"], sep=float(sep), sigma=t["sigma"])
        for seed in seeds:
            seed_everything(seed)
            mix = make_mixture(spec, get_rng(seed))
            w, _ = bayes_init(mix.means, spec.sigma)
            z0 = mix.X @ w.T
            probs[(sep, seed)] = S.SampleProblem(z0, 0.0)
            for lam in st["lam_list"]:
                prob = S.SampleProblem(z0, lam)
                b_star, gnorm = S.find_stationary(prob)
                h_an = prob.hess(b_star)
                h_fd = S.fd_hessian(prob, b_star)
                h_t = S.torch_hessian(prob, b_star)
                ev = S.perp_eigs(h_an)
                pop_row = popdf[(popdf.sep == sep) & (popdf.lam == lam) & (popdf.mc_seed == pp["mc_seeds"][0])].iloc[0]
                gap = float(b_star[0] - b_star[1:].mean())
                # [assumption] b* counts as "the balanced point" only if it is a true stationary point within
                # |gap| < 2 logits of b = 0; otherwise Newton found some other (collapsed/flat) stationary point.
                near = bool(gnorm < 1e-9 and abs(gap) < 2.0)
                ev0 = S.perp_eigs(prob.hess(np.zeros(K)))     # curvature at exactly b=0 (always defined)
                stable = bool(near and ev.min() > 1e-8)   # tiny |eig| = marginal/degenerate, not 'stable'
                eta_s = 2.0 / ev.max() if ev.max() > 0 and stable else float("nan")
                row = {"sep": sep, "seed": seed, "lam": lam, "grad_norm_at_b_star": gnorm,
                       "b_star_gap": gap, "near_balanced": near,
                       "eig_min_at0": float(ev0.min()), "eig_max_at0": float(ev0.max()),
                       "eig_min_perp": float(ev.min()), "eig_max_perp": float(ev.max()),
                       "alpha_J_pop": pop_row.alpha_J, "eta_star_pop": pop_row.eta_star_pop,
                       "hess_fd_maxerr": float(np.abs(h_fd - h_an).max()),
                       "hess_torch_maxerr": float(np.abs(h_t - h_an).max()) if h_t is not None else float("nan"),
                       "locally_stable_for_some_eta": stable, "eta_star_sample": eta_s,
                       "eta_star_numeric": S.eta_star_numeric(prob, b_star, eta_s) if stable else float("nan")}
                for f in st["eta_check_factors"]:
                    row[f"rho_map_{f}"] = S.map_spectral_radius(h_fd, f * eta_s) if stable else float("nan")
                srow.append(row)
                eta_sim = min(st["sim_eta_max"], 0.5 * eta_s) if stable else st["sim_eta_max"]
                res = sim_from_zero(prob, eta_sim, st["sim_steps"])
                zrow.append({"sep": sep, "seed": seed, "lam": lam, "eta": eta_sim, "steps": st["sim_steps"],
                             "collapsed": is_collapsed(res["H_bar_final"], K, tau), **res})
            log.info("sample stability done: sep=%s seed=%s", sep, seed)
    stab, sim = pd.DataFrame(srow), pd.DataFrame(zrow)
    stab.to_csv(run_dir / "stability.csv", index=False)
    sim.to_csv(run_dir / "sim_from_zero.csv", index=False)

    # ---- D) sample-level reduced-flow root vs Phase 1 simulated thresholds (optional)
    p1 = Path(cfg["compare"]["phase1_thresholds"])
    rrow: list[dict] = []
    if p1.exists():
        thr = pd.read_csv(p1)
        thr = thr[thr.budget == "default"]
        for (sep, seed), prob in probs.items():
            sel = thr[(np.isclose(thr.sep, sep)) & (thr.seed == seed)]
            if sel.empty:
                continue
            m = sep**2 / t["sigma"] ** 2

            def f1(g: float) -> float:
                b = np.zeros(K); b[0] = g
                bd = D.bias_flow_from_logp(prob.logp(b), 0.0)
                return float(bd[0] - bd[1:].mean())

            def spread(g: float) -> float:                 # asymmetry among classes 1..K-1 vs drift
                b = np.zeros(K); b[0] = g
                bd = D.bias_flow_from_logp(prob.logp(b), 0.0)
                return float(np.ptp(bd[1:]))

            rt = D.gap_root(f1, m, 1.5, 121, "last", xtol=1e-5)
            rrow.append({"sep": sep, "seed": seed, "phase1_delta_star": float(sel.delta_star.iloc[0]),
                         "phase1_status": sel.status.iloc[0], "sample_root_last": rt.g_c,
                         "sample_root_status": rt.status, "n_up_crossings": rt.n_up_crossings,
                         "asym_over_drift_at_root": (spread(rt.g_c) / max(abs(f1(rt.g_c + 0.5)), 1e-12)
                                                     if rt.status == "ok" else float("nan"))})
    else:
        log.warning("Phase 1 thresholds not found at %s: comparison skipped", p1)
    pd.DataFrame(rrow).to_csv(run_dir / "sample_roots.csv", index=False)

    # ---- summary (all numbers derived from the CSVs above)
    ok = stab[stab.locally_stable_for_some_eta]
    nb = stab[stab.near_balanced]
    pred_stable = {(r.sep, r.lam): r.alpha_J > 0 for r in popdf[popdf.mc_seed == pp["mc_seeds"][0]].itertuples()}
    sim["pred_stable"] = [pred_stable[(r.sep, r.lam)] for r in sim.itertuples()]
    summary = {
        "n_rows": int(len(stab)), "n_near_balanced": int(len(nb)),
        "n_near_balanced_stable": int(len(ok)),
        "n_near_balanced_unstable": int((nb.eig_min_perp <= 1e-8).sum()),
        "n_no_nearby_stationary_point": int((~stab.near_balanced).sum()),
        "frac_eig_min_at0_positive_by_sep_lam": {f"sep={k[0]:g},lam={k[1]:g}": float(v) for k, v in
            (stab.assign(pos=stab.eig_min_at0 > 1e-8).groupby(["sep", "lam"]).pos.mean()).items()},
        "hess_fd_maxerr_overall": float(stab.hess_fd_maxerr.max()),
        "hess_torch_maxerr_overall": (float(stab.hess_torch_maxerr.max()) if stab.hess_torch_maxerr.notna().any() else None),
        "max_grad_norm_at_b_star": float(stab.grad_norm_at_b_star.max()),
        "eta_numeric_vs_sample_relerr_max": float((abs(ok.eta_star_numeric - ok.eta_star_sample) / ok.eta_star_sample).max()),
        "eta_numeric_vs_sample_relerr_n_nan": int(ok.eta_star_numeric.isna().sum()),
        "eta_sample_vs_pop_ratio_mean_std": [float((ok.eta_star_sample / ok.eta_star_pop).mean()),
                                              float((ok.eta_star_sample / ok.eta_star_pop).std(ddof=0))],
        "rho_map_median_by_factor": {f: float(ok[f"rho_map_{f}"].median()) for f in st["eta_check_factors"]},
        "stability_prediction_vs_sim_from_zero": {
            "n": int(len(sim)), "agree": int(((~sim.collapsed) == sim.pred_stable).sum())},
        "mc_spread_g_c_lam0_max": float(popdf[popdf.lam == 0.0].groupby("sep").g_c.agg(lambda x: x.max() - x.min()).max()),
    }
    with RunLogger(run_dir) as rl:
        rl.write_summary(summary)
    log.info("summary written")
    print(run_dir)


if __name__ == "__main__":
    main()
