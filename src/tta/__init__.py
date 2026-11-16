from src.tta.base import TTAMethod
from src.tta.source import BNAdapt, Source
from src.tta.tent import Tent

__all__ = ["TTAMethod", "Source", "BNAdapt", "Tent", "build_method"]


def build_method(name: str, model, tta_cfg: dict):
    """Factory used by scripts/run_stream.py. `tta_cfg` is the resolved `tta.<name>` section (may be empty)."""
    if name == "source":
        return Source(model)
    if name == "bn_adapt":
        return BNAdapt(model)
    if name == "tent":
        return Tent(model, lr=float(tta_cfg.get("lr", 1e-3)), betas=tuple(tta_cfg.get("betas", (0.9, 0.999))),
                    weight_decay=float(tta_cfg.get("weight_decay", 0.0)), steps=int(tta_cfg.get("steps", 1)),
                    optimizer=str(tta_cfg.get("optimizer", "adam")))
    raise ValueError(f"Unknown method {name!r} (Phase 3 has: source, bn_adapt, tent)")
