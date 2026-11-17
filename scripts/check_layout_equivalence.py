"""Does changing the input layout (as_is channels_last strides vs contiguous NCHW) change the RESULTS?

Runs each method on the SAME first --steps batches of the real stream (same seed, same model copy) under each layout and
compares them. Bitwise-identical logits are NOT expected (different memory layout => different cuDNN kernels / summation order).
Reported per method (a = as_is, b = nchw):
  max|dlogit|          largest absolute logit difference over all steps and samples
  pred diffs           number of samples whose predicted class differs (out of steps x batch_size)
  err a / err b        mean online error of each layout; d_err = b - a
  steps differ         number of batches whose error differs AT ALL (comparison is EXACT, zero tolerance: error per batch is
                       count/batch_size, so 'equal' means exactly the same number of wrong predictions in that batch)
  max|d_err step|      largest per-batch error difference
  max|dparam|          largest absolute difference in the final model state (TENT's BN gamma/beta)
--control adds a row 'nchw vs nchw(rerun)': same layout, same seed, run twice. If that row is NOT all zeros, the pipeline is
not bit-reproducible by itself, and layout is not the only source of differences.
--segments prints mean error per corruption segment for both layouts (where do the trajectories drift apart?).
Results are saved to results/check_layout_equivalence/<experiment.name>_steps<N>_seed<S>.json (so the numbers have a log).
Usage:  python scripts/check_layout_equivalence.py --config configs/stream_tent.yaml --steps 750 --control --segments
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.stream import batches_per_segment, iter_stream, spec_from_cfg
from src.data.tensors import to_tensor
from src.models.loader import load_source_model
from src.tta import build_method
from src.utils.config import load_config
from src.utils.seeding import seed_everything


def run(cfg: dict, base, method: str, layout: str, steps: int, seed: int, dev: torch.device):
    seed_everything(seed, deterministic=bool(cfg["compute"]["deterministic"]))
    tta = build_method(method, copy.deepcopy(base), cfg.get("tta", {}).get(method, {}))
    spec = spec_from_cfg(cfg, seed)
    spec = type(spec)(**{**spec.__dict__, "max_steps": steps})
    logits, errs = [], []
    for batch in iter_stream(spec):
        lg = tta(to_tensor(batch.x, dev, layout)).float().cpu().numpy()
        logits.append(lg)
        errs.append(float((lg.argmax(1) != batch.y).mean()))
    params = {k: v.detach().cpu().double() for k, v in tta.model.state_dict().items() if v.dtype.is_floating_point}
    return np.stack(logits), np.array(errs), params


def compare(a, b) -> dict:
    (la, ea, pa), (lb, eb, pb) = a, b
    d_err = np.abs(ea - eb)
    return {"steps": int(len(ea)), "samples": int(la.shape[0] * la.shape[1]),
            "max_abs_dlogit": float(np.abs(la - lb).max()), "pred_diffs": int((la.argmax(2) != lb.argmax(2)).sum()),
            "err_a": float(ea.mean()), "err_b": float(eb.mean()), "d_err": float(eb.mean() - ea.mean()),
            "steps_differ": int((d_err > 0).sum()), "max_abs_step_err_diff": float(d_err.max()),
            "max_abs_dparam": max((float((pa[k] - pb[k]).abs().max()) for k in pa), default=0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream_tent.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["source", "bn_adapt", "tent"])
    ap.add_argument("--control", action="store_true", help="also compare nchw with an identical nchw rerun")
    ap.add_argument("--segments", action="store_true", help="print per-corruption-segment mean error for both layouts")
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, model_info = load_source_model(cfg["model"], dev)
    seg = batches_per_segment(spec_from_cfg(cfg, args.seed))
    rows, segs = [], {}
    for m in args.methods:
        a, b = run(cfg, base, m, "as_is", args.steps, args.seed, dev), run(cfg, base, m, "nchw", args.steps, args.seed, dev)
        rows.append({"method": m, "pair": "as_is vs nchw", **compare(a, b)})
        if args.control:
            rows.append({"method": m, "pair": "nchw vs nchw (control)",
                         **compare(b, run(cfg, base, m, "nchw", args.steps, args.seed, dev))})
        if args.segments:
            n_seg = len(a[1]) // seg
            segs[m] = {"as_is": [float(a[1][i * seg:(i + 1) * seg].mean()) for i in range(n_seg)],
                       "nchw": [float(b[1][i * seg:(i + 1) * seg].mean()) for i in range(n_seg)]}
    hdr = (f"{'method':<9}{'pair':<24}{'steps':>6}{'max|dlogit|':>12}{'pred diffs':>11}{'/samples':>9}{'err a':>9}{'err b':>9}"
           f"{'d_err':>9}{'steps differ':>13}{'max|d_err|':>11}{'max|dparam|':>12}")
    print(hdr)
    for r in rows:
        print(f"{r['method']:<9}{r['pair']:<24}{r['steps']:>6}{r['max_abs_dlogit']:>12.3e}{r['pred_diffs']:>11}{r['samples']:>9}"
              f"{r['err_a']:>9.5f}{r['err_b']:>9.5f}{r['d_err']:>+9.5f}{r['steps_differ']:>13}{r['max_abs_step_err_diff']:>11.4f}"
              f"{r['max_abs_dparam']:>12.3e}")
    for m, d in segs.items():
        print(f"\nper-segment mean error, {m} (segment = {seg} batches):")
        print("  seg  as_is    nchw     diff")
        for i, (x, y) in enumerate(zip(d["as_is"], d["nchw"])):
            print(f"  {i:>3}  {x:.4f}  {y:.4f}  {y - x:+.4f}")
    out = Path(cfg["experiment"]["results_dir"]) / "check_layout_equivalence"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg['experiment']['name']}_steps{args.steps}_seed{args.seed}.json"
    path.write_text(json.dumps({"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "model": model_info,
                                "deterministic": cfg["compute"]["deterministic"], "args": vars(args), "rows": rows,
                                "segments": segs}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
