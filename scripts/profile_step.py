"""Where does the time per step go? Times each component of a stream step on one fixed real batch.

Usage (run each line; compare the printed tables):
    python scripts/profile_step.py --config configs/stream_tent.yaml --deterministic true
    python scripts/profile_step.py --config configs/stream_tent.yaml --deterministic false
    python scripts/profile_step.py --config configs/stream_tent.yaml --deterministic false --benchmark
GPU work is timed with cuda synchronisation after a warm-up, so one-off start-up costs are excluded (and reported
separately as the first-iteration time). nvidia-smi clocks/temperature/throttle flags are printed before and after,
to expose laptop-GPU throttling. Nothing here is a research result.
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.cifar10c import load_corruption
from src.data.tensors import to_tensor
from src.metrics.collapse import batch_metrics
from src.metrics.entropy import mean_entropy
from src.models.loader import load_source_model
from src.tta import BNAdapt, Source, Tent
from src.utils.config import load_config
from src.utils.seeding import seed_everything


def smi(tag: str) -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode().strip()
        print(f"[nvidia-smi {tag}] sm_clock, sm_clock_max, power, temp, throttle_reasons_bitmask = {out}")
    except Exception:
        print(f"[nvidia-smi {tag}] not available")


def timeit(fn, n: int = 20, warmup: int = 5) -> tuple[float, float, float]:
    """Returns (first_call_s, median_s, p90_s) with GPU synchronisation."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t = time.perf_counter(); fn(); sync(); first = time.perf_counter() - t
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return first, float(np.median(ts)), float(np.percentile(ts, 90))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream_tent.yaml")
    ap.add_argument("--deterministic", choices=["true", "false"], default="true")
    ap.add_argument("--benchmark", action="store_true", help="enable cudnn.benchmark (autotuner)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    det = args.deterministic == "true"
    seed_everything(0, deterministic=det)
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"deterministic={det} benchmark={args.benchmark} device={dev} torch={torch.__version__}")
    smi("before")
    base, _ = load_source_model(cfg["model"], dev)
    bs = cfg["stream"]["batch_size"]
    x_np, y_np = load_corruption(cfg["data"]["root"], "gaussian_noise", 5)
    xb_np, yb_np = x_np[:bs], y_np[:bs]
    layout = cfg["compute"].get("input_layout", "as_is")
    x = to_tensor(xb_np, dev, layout)

    src, bna = Source(copy.deepcopy(base)), BNAdapt(copy.deepcopy(base))
    tent = Tent(copy.deepcopy(base), lr=1e-3)
    tent_nostep = Tent(copy.deepcopy(base), lr=1e-3)

    def tent_fwd_bwd() -> None:
        loss = mean_entropy(tent_nostep.model(x))
        tent_nostep.optimizer.zero_grad(set_to_none=True)
        loss.backward()

    logits_np = src(x).float().cpu().numpy()
    rows = {
        "host: fancy-index a batch (numpy)": lambda: x_np[np.sort(np.random.permutation(len(x_np))[:bs])],
        "host->GPU: uint8 -> float tensor": lambda: to_tensor(xb_np, dev, layout),
        "Source forward (eval BN, no_grad)": lambda: src(x),
        "BNAdapt forward (batch-stat BN, no_grad)": lambda: bna(x),
        "TENT forward+backward (no optimizer step)": tent_fwd_bwd,
        "TENT full step (fwd+bwd+Adam)": lambda: tent(x),
        "GPU->host + float64 metrics": lambda: batch_metrics(src(x).float().cpu().numpy(), yb_np),
    }
    print(f"{'component':<45}{'first call':>12}{'median':>10}{'p90':>10}   (seconds, batch={bs})")
    for name, fn in rows.items():
        f, med, p90 = timeit(fn)
        print(f"{name:<45}{f:>12.4f}{med:>10.4f}{p90:>10.4f}")
    smi("after")


if __name__ == "__main__":
    main()
