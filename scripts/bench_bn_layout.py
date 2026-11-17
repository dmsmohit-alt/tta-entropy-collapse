"""Why is batch-statistics BatchNorm so much slower than eval-mode BatchNorm? Two parts.

  micro : F.batch_norm on the ResNet-18 activation shapes (batch from config), for
          layout in {NCHW, channels_last} x cudnn {on, off} x mode {eval, batch-stat (running stats None), batch-stat fwd+bwd}.
  model : Source / BNAdapt / TENT timings on the real network for 4 input/model layouts, plus the strides of the input
          tensor produced by the current to_tensor (so we SEE whether it is channels_last).

Usage:  python scripts/bench_bn_layout.py --config configs/stream_tent.yaml [--part micro|model|all]
Writes results/bench_bn_layout/{micro,model}.csv. Timings use cuda synchronisation after warm-up. Nothing here is a research result.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.cifar10c import load_corruption
from src.data.tensors import to_tensor
from src.metrics.entropy import mean_entropy
from src.models.loader import load_source_model
from src.tta import BNAdapt, Source, Tent
from src.utils.config import load_config

SHAPES = [(64, 32, 32), (128, 16, 16), (256, 8, 8), (512, 4, 4)]   # BN activation shapes in the CIFAR ResNet-18


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median_ms(fn, n: int = 30, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return 1000.0 * float(np.median(ts))


def micro(bs: int, dev: torch.device) -> pd.DataFrame:
    rows = []
    for C, H, W in SHAPES:
        base = torch.randn(bs, C, H, W, device=dev)
        for layout in ("nchw", "channels_last"):
            x = base if layout == "nchw" else base.contiguous(memory_format=torch.channels_last)
            assert x.is_contiguous(memory_format=torch.channels_last if layout == "channels_last" else torch.contiguous_format)
            w, b = torch.ones(C, device=dev), torch.zeros(C, device=dev)
            rm, rv = torch.zeros(C, device=dev), torch.ones(C, device=dev)
            wg, bg = torch.ones(C, device=dev, requires_grad=True), torch.zeros(C, device=dev, requires_grad=True)

            def fwdbwd() -> None:
                F.batch_norm(x, None, None, wg, bg, True, 0.1, 1e-5).sum().backward()

            for cudnn_on in (True, False):
                with torch.backends.cudnn.flags(enabled=cudnn_on, benchmark=False, deterministic=False):
                    ev = median_ms(lambda: F.batch_norm(x, rm, rv, w, b, False, 0.1, 1e-5))
                    bs_ = median_ms(lambda: F.batch_norm(x, None, None, w, b, True, 0.1, 1e-5))
                    fb = median_ms(fwdbwd)
                rows.append({"shape": f"{bs}x{C}x{H}x{W}", "layout": layout, "cudnn": cudnn_on, "eval_ms": ev,
                             "batchstat_ms": bs_, "batchstat_fwdbwd_ms": fb, "batchstat_over_eval": bs_ / ev})
    return pd.DataFrame(rows)


def model_bench(cfg: dict, dev: torch.device) -> pd.DataFrame:
    base, _ = load_source_model(cfg["model"], dev)
    bs = cfg["stream"]["batch_size"]
    x_np, _ = load_corruption(cfg["data"]["root"], "gaussian_noise", 5)
    x_np = x_np[:bs]
    x_asis, x_nchw = to_tensor(x_np, dev, "as_is"), to_tensor(x_np, dev, "nchw")
    print(f"as_is : strides={x_asis.stride()} contiguous={x_asis.is_contiguous()} "
          f"channels_last={x_asis.is_contiguous(memory_format=torch.channels_last)}")
    print(f"nchw  : strides={x_nchw.stride()} contiguous={x_nchw.is_contiguous()} "
          f"channels_last={x_nchw.is_contiguous(memory_format=torch.channels_last)}")
    print(f"values identical: {bool(torch.equal(x_asis, x_nchw))}")
    variants = {"as_is input, NCHW model": (x_asis, False), "nchw input, NCHW model": (x_nchw, False),
                "nchw input, channels_last model": (x_nchw, True), "as_is input, channels_last model": (x_asis, True)}
    rows = []
    for name, (x, model_cl) in variants.items():
        def make(cls, **kw):
            m = cls(copy.deepcopy(base), **kw)
            if model_cl:
                m.model.to(memory_format=torch.channels_last)
            return m
        src, bna, tent, tent_nostep = make(Source), make(BNAdapt), make(Tent, lr=1e-3), make(Tent, lr=1e-3)

        def fwdbwd() -> None:
            loss = mean_entropy(tent_nostep.model(x))
            tent_nostep.optimizer.zero_grad(set_to_none=True)
            loss.backward()

        rows.append({"variant": name, "source_fwd_ms": median_ms(lambda: src(x), 15, 5),
                     "bnadapt_fwd_ms": median_ms(lambda: bna(x), 15, 5),
                     "tent_fwdbwd_ms": median_ms(fwdbwd, 15, 5), "tent_step_ms": median_ms(lambda: tent(x), 15, 5)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream_tent.yaml")
    ap.add_argument("--part", choices=["micro", "model", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["experiment"]["results_dir"]) / "bench_bn_layout"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev} torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        if args.part in ("micro", "all"):
            df = micro(cfg["stream"]["batch_size"], dev)
            df.to_csv(out / "micro.csv", index=False)
            print("\n== F.batch_norm micro-benchmark (median ms) ==\n", df.to_string(index=False))
        if args.part in ("model", "all"):
            df = model_bench(cfg, dev)
            df.to_csv(out / "model.csv", index=False)
            print("\n== full network (median ms per batch) ==\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
