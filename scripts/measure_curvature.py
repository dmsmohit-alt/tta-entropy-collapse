"""Measure Hessian extremes of the TTA loss on BN parameters at the SOURCE weights (BN in batch-statistics mode).

Usage:
    python scripts/measure_curvature.py --config configs/stream_select_sev3.yaml --lams 0 1 --n-segments 5
Writes results/curvature/<experiment.name>/curvature.csv and prints eta*_GD = 2/lambda_max per (segment, lambda).

Use: pick the SGD arm's learning rate on the SEVERITY-3 selection stream (not the reported severity-5 stream):
    eta = c * min over segments of 2/lambda_max, with c = 0.5 (our choice, an assumption; c < 1 for margin).
Limitations (also in README): measured once, at the source parameters, on one batch per segment; the Hessian moves as
the BN parameters adapt, and a single batch is a noisy estimate. Needs double backward: on 4 GB VRAM use
--batch-size 100 if you run out of memory (this changes the BN batch statistics, so say so when you report it).
"""
from __future__ import annotations

import argparse
import copy
import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd
import torch

from src.data.stream import batches_per_segment, iter_stream, spec_from_cfg
from src.data.tensors import to_tensor
from src.metrics.entropy import info_max_loss, mean_entropy
from src.models.bn import configure_model_for_tta
from src.models.loader import load_source_model
from src.tta.curvature import hessian_extremes, hessian_extremes_lanczos
from src.utils.config import load_config
from src.utils.seeding import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--n-segments", type=int, default=5, help="first batch of each of the first N corruption segments")
    ap.add_argument("--method", choices=["lanczos", "power"], default="lanczos",
                    help="power = old shifted power iteration; UNRELIABLE when |lam_min| >> lam_max (see src/utils/lanczos.py)")
    ap.add_argument("--n-iter", type=int, default=50, help="power method only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    if a.batch_size:
        cfg["stream"]["batch_size"] = a.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["compute"]["device"] != "cpu" else "cpu")
    seed_everything(a.seed, deterministic=bool(cfg["compute"]["deterministic"]))
    base, info = load_source_model(cfg["model"], device)
    spec = spec_from_cfg(cfg, a.seed)
    per_seg = batches_per_segment(spec)
    layout = cfg["compute"].get("input_layout", "nchw")
    rows = []
    for batch in iter_stream(spec):
        seg = batch.step // per_seg
        if seg >= a.n_segments:
            break
        if batch.step % per_seg:
            continue
        x = to_tensor(batch.x, device, layout)
        for lam in a.lams:
            model = configure_model_for_tta(copy.deepcopy(base))
            lf = mean_entropy if lam == 0 else partial(info_max_loss, lam=lam)
            e = (hessian_extremes_lanczos(model, lf, x, seed=a.seed) if a.method == "lanczos"
                 else hessian_extremes(model, lf, x, n_iter=a.n_iter, seed=a.seed))
            eta_star = 2.0 / e.lam_max if e.lam_max > 0 else float("nan")
            rows.append({"segment": seg, "corruption": batch.corruption, "severity": spec.severity, "batch_size": len(batch.y),
                         "lam": lam, "method": a.method, "lam_max": e.lam_max, "lam_min": e.lam_min, "eta_star_gd": eta_star,
                         "converged_max": e.converged_max, "converged_min": e.converged_min, "n_hvp": e.n_hvp})
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
    df = pd.DataFrame(rows)
    out = REPO / cfg["experiment"]["results_dir"] / "curvature" / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "curvature.csv", index=False)
    summ = df.groupby("lam").agg(lam_max_max=("lam_max", "max"), lam_min_min=("lam_min", "min"),
                                 eta_star_min=("eta_star_gd", "min"), all_converged=("converged_max", "all"))
    print(summ.to_string())
    print(f"\nwritten: {out / 'curvature.csv'}\n(source checkpoint sha256: {info.get('sha256')})")


if __name__ == "__main__":
    main()
