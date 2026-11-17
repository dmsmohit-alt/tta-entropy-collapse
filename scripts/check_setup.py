"""Sanity-check data + model BEFORE any long run: source-model error per corruption and severity.

Usage:  python scripts/check_setup.py --config configs/stream_tent.yaml [key=value ...]
What it checks (prints a table, saves JSON; it does not claim any particular accuracy):
  * the .npy layout loads (shape/dtype asserted in src/data/cifar10c.py);
  * the source model's error at severity 5 is HIGHER than at severity 1 for each tested corruption,
    which would be false if the severity slicing assumption were wrong (e.g. wrong order of the stacked severities);
  * the model runs on the chosen device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.cifar10c import load_corruption
from src.data.tensors import to_tensor
from src.models.loader import load_source_model
from src.utils.config import load_config

TEST_CORRUPTIONS = ("gaussian_noise", "fog", "brightness")
SEVERITIES = (1, 3, 5)


@torch.no_grad()
def source_error(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, layout: str = "as_is",
                 bs: int = 500) -> float:
    wrong = 0
    for i in range(0, len(x), bs):
        xb = to_tensor(x[i:i + bs], device, layout)
        wrong += int((model(xb).argmax(1).cpu().numpy() != y[i:i + bs]).sum())
    return wrong / len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream_tent.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = torch.device("cuda" if cfg["compute"]["device"] == "auto" and torch.cuda.is_available()
                       else ("cpu" if cfg["compute"]["device"] == "auto" else cfg["compute"]["device"]))
    model, info = load_source_model(cfg["model"], dev)
    print(f"device={dev} model={info}")
    res: dict[str, dict[int, float]] = {}
    for c in TEST_CORRUPTIONS:
        res[c] = {}
        for s in SEVERITIES:
            x, y = load_corruption(cfg["data"]["root"], c, s)
            res[c][s] = source_error(model, x, y, dev, cfg["compute"].get("input_layout", "as_is"))
    print(f"{'corruption':<16}" + "".join(f"sev{s:<7}" for s in SEVERITIES))
    for c, d in res.items():
        print(f"{c:<16}" + "".join(f"{d[s]:<10.4f}" for s in SEVERITIES))
    ok = all(d[SEVERITIES[0]] < d[SEVERITIES[-1]] for d in res.values())
    print("severity ordering check (sev1 error < sev5 error for all):", "OK" if ok else "FAILED: check the data layout!")
    out = Path(cfg["experiment"]["results_dir"]) / "check_setup"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_setup.json").write_text(json.dumps({"model": info, "source_error": res, "ordering_ok": ok}, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
