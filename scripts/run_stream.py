"""Run TTA methods over a continual CIFAR-10-C stream and log per-step metrics.

Usage:
    python scripts/run_stream.py --config configs/stream_tent.yaml [key=value ...]
    python scripts/run_stream.py --config configs/stream_smoke.yaml           # tiny plumbing run
Every (method, seed) writes results/<experiment.name>/<method>/<confighash>_seed<N>/ with
config.yaml, metrics.csv (one row per batch), summary.json, run.log. All methods see the IDENTICAL
stream for a given seed (the stream depends only on the seed, not on the method).
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.data.stream import iter_stream, spec_from_cfg, steps_per_cycle, total_steps
from src.metrics.collapse import CollapseDetector, EmaMarginal, batch_metrics
from src.models.loader import load_source_model
from src.tta import build_method
from src.utils.config import load_config, make_run_dir, save_config
from src.utils.run_logger import RunLogger, get_text_logger
from src.utils.seeding import seed_everything


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def to_tensor(x_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 NHWC numpy -> float32 NCHW tensor in [0, 1] on `device`."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def run_one(cfg: dict, method: str, seed: int, base_model: torch.nn.Module, model_info: dict,
            device: torch.device) -> dict:
    run_cfg = {k: cfg[k] for k in ("data", "stream", "model", "metrics", "compute") if k in cfg}
    run_cfg["experiment"] = {"name": f"{cfg['experiment']['name']}/{method}",
                             "results_dir": cfg["experiment"]["results_dir"]}
    run_cfg["method"] = method
    run_cfg["tta"] = cfg.get("tta", {}).get(method, {})
    run_dir = make_run_dir(run_cfg, seed)                 # folder name = hash of config WITHOUT the seed + _seedN
    save_config({**run_cfg, "seed": seed}, run_dir)       # saved config does include the seed
    if cfg["experiment"].get("skip_existing", False) and (run_dir / "summary.json").exists():
        print(f"skip (exists): {run_dir}")
        return json.loads((run_dir / "summary.json").read_text())

    log = get_text_logger(f"run.{method}.{seed}", run_dir / "run.log")
    seed_everything(seed, deterministic=bool(cfg["compute"]["deterministic"]))
    tta = build_method(method, copy.deepcopy(base_model), run_cfg["tta"])
    spec = spec_from_cfg(cfg, seed)
    n_steps = total_steps(spec)
    mcfg = cfg["metrics"]
    ema = EmaMarginal(float(mcfg["ema_decay"]))
    det = CollapseDetector(float(mcfg["collapse_top_frac"]), int(mcfg["collapse_patience"]))
    err_sum: dict[int, float] = {}
    cnt: dict[int, int] = {}
    topfrac_by_cycle: dict[int, list[float]] = {}
    max_top, last_h_ema, last_h_batch, diverged_step = 0.0, float("nan"), float("nan"), None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log.info("%s seed=%d | %d steps (%d per cycle) | trainable params %d | device %s",
             method, seed, n_steps, steps_per_cycle(spec), tta.n_adapted_params, device)
    t0 = time.time()
    with RunLogger(run_dir) as rl:
        for batch in iter_stream(spec):
            logits = tta(to_tensor(batch.x, device))
            lg = logits.float().cpu().numpy()
            if not np.isfinite(lg).all():                 # numerical blow-up: record and stop, do not hide it
                diverged_step = batch.step
                log.warning("non-finite logits at step %d: stopping this run", batch.step)
                break
            m, pbar = batch_metrics(lg, batch.y)
            h_ema = ema.update(pbar)
            det.update(batch.step, m["top_frac"])
            rl.log_step(batch.step, {"cycle": batch.cycle, "corruption": batch.corruption, **m,
                                     "H_bar_ema": h_ema, "collapsed_flag": int(det.collapsed)})
            n = len(batch.y)
            err_sum[batch.cycle] = err_sum.get(batch.cycle, 0.0) + m["online_err"] * n
            cnt[batch.cycle] = cnt.get(batch.cycle, 0) + n
            topfrac_by_cycle.setdefault(batch.cycle, []).append(m["top_frac"])
            max_top, last_h_ema, last_h_batch = max(max_top, m["top_frac"]), h_ema, m["H_bar_batch"]
            if batch.step % max(1, n_steps // 20) == 0:
                log.info("step %d/%d cycle %d %-18s err=%.3f H_bar_ema=%.3f top_frac=%.2f",
                         batch.step, n_steps, batch.cycle, batch.corruption, m["online_err"], h_ema, m["top_frac"])
        seconds = time.time() - t0
        cycles = sorted(cnt)
        err_by_cycle = [err_sum[c] / cnt[c] for c in cycles]
        summary = {
            "method": method, "seed": seed, "steps_done": int(sum(cnt.values()) // max(1, spec.batch_size)),
            "steps_planned": n_steps, "diverged_step": diverged_step,
            "overall_err": float(sum(err_sum.values()) / max(1, sum(cnt.values()))),
            "err_by_cycle": err_by_cycle,
            "err_last_cycle": err_by_cycle[-1] if err_by_cycle else None,
            "final_acc_last_cycle": (1.0 - err_by_cycle[-1]) if err_by_cycle else None,
            "max_top_frac": max_top,
            "mean_top_frac_last_cycle": float(np.mean(topfrac_by_cycle[cycles[-1]])) if cycles else None,
            "final_H_bar_ema": last_h_ema, "final_H_bar_batch": last_h_batch,
            "collapsed": det.collapsed, "time_to_collapse_start": det.start_step, "time_to_collapse_detect": det.detect_step,
            "collapse_rule": {"top_frac_gt": mcfg["collapse_top_frac"], "consecutive_batches": mcfg["collapse_patience"]},
            "n_adapted_params": tta.n_adapted_params, "seconds": seconds,
            "peak_cuda_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
            "model": model_info, "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_commit(),
        }
        rl.write_summary(summary)
    log.info("done: overall_err=%.4f last_cycle_err=%s collapsed=%s start=%s", summary["overall_err"],
             summary["err_last_cycle"], summary["collapsed"], summary["time_to_collapse_start"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream_tent.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. stream.num_cycles=3 methods=[tent]")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = pick_device(cfg["compute"]["device"])
    base_model, model_info = load_source_model(cfg["model"], device)
    for method in cfg["methods"]:
        for seed in cfg["experiment"]["seeds"]:
            run_one(cfg, method, int(seed), base_model, model_info, device)


if __name__ == "__main__":
    main()
