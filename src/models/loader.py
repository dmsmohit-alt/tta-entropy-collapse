"""Load the source (pre-trained) model.

Two sources:
  checkpoint : our own standard-trained CIFAR ResNet-18 (scripts/train_source.py). Default for development
               on a 4 GB GPU. Standard training (random crop + flip), NOT corruption-robust training.
  robustbench: a model from the RobustBench zoo, e.g. name="Standard" (WRN-28-10, the usual benchmark model)
               with threat_model="corruptions". Needs `pip install git+https://github.com/RobustBench/robustbench.git`.
               WRN-28-10 is heavy for 4 GB VRAM (use a Colab T4). Note "Modas2021PRIMEResNet18" is a ResNet-18
               trained with PRIME augmentation (corruption-robust), so it is NOT comparable to a standard ResNet-18.
All models take [0, 1] NCHW float input and return logits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.resnet_cifar import ResNet18CIFAR


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_model(model_cfg: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Return (model in eval mode on `device`, info dict for the run summary)."""
    source = model_cfg["source"]
    if source == "checkpoint":
        path = Path(model_cfg["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train it with scripts/train_source.py or set model.source=robustbench.")
        model = ResNet18CIFAR()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(blob["model"] if "model" in blob else blob)
        info = {"source": "checkpoint", "arch": "ResNet18CIFAR", "path": str(path), "sha256": file_sha256(path),
                "clean_test_acc_at_training": blob.get("test_acc") if isinstance(blob, dict) else None}
    elif source == "robustbench":
        from robustbench.utils import load_model  # lazy: optional dependency

        model = load_model(model_name=model_cfg["name"], model_dir=model_cfg.get("models_dir", "models"),
                           dataset="cifar10", threat_model="corruptions")
        info = {"source": "robustbench", "name": model_cfg["name"], "threat_model": "corruptions"}
    else:
        raise ValueError(f"model.source must be 'checkpoint' or 'robustbench', got {source!r}")
    return model.to(device).eval(), info
