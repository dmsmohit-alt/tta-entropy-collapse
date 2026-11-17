from functools import partial

from src.metrics.entropy import info_max_loss
from src.tta.base import TTAMethod
from src.tta.reset import PeriodicReset
from src.tta.source import BNAdapt, Source
from src.tta.tent import Tent

__all__ = ["TTAMethod", "Source", "BNAdapt", "Tent", "PeriodicReset", "build_method"]


def build_method(name: str, model, tta_cfg: dict):
    """Factory used by scripts/run_stream.py. `tta_cfg` is the resolved `tta.<name>` section (may be empty).

    `name` is just a label (it names the results folder); the algorithm is `tta_cfg["kind"]`, default = `name`.
    kind "tent" accepts:
      lam         (default 0)   0 = plain entropy (TENT); >0 = entropy - lam * H(p_bar)  ("TENT + marginal entropy")
      reset_every (default None) int T -> wrap in PeriodicReset (RDumb-style baseline)
      optimizer adam|sgd, lr (REQUIRED, no silent default for sgd), betas, momentum, weight_decay, steps
    """
    kind = tta_cfg.get("kind", name)
    if kind == "source":
        return Source(model)
    if kind == "bn_adapt":
        return BNAdapt(model)
    if kind == "tent":
        optimizer = str(tta_cfg.get("optimizer", "adam"))
        lr = tta_cfg.get("lr", 1e-3 if optimizer == "adam" else None)
        if lr is None:
            raise ValueError(f"tta.{name}.lr is not set. For the SGD arm choose lr < 2/lambda_max from "
                             "scripts/measure_curvature.py and pass it explicitly (override tta.<name>.lr=...).")
        lam = float(tta_cfg.get("lam", 0.0))
        method = Tent(model, lr=float(lr), betas=tuple(tta_cfg.get("betas", (0.9, 0.999))),
                      weight_decay=float(tta_cfg.get("weight_decay", 0.0)), steps=int(tta_cfg.get("steps", 1)),
                      optimizer=optimizer, momentum=float(tta_cfg.get("momentum", 0.0)),
                      loss_fn=partial(info_max_loss, lam=lam) if lam != 0.0 else None)  # lam=0 -> exactly Phase 3's TENT
        period = tta_cfg.get("reset_every")
        return PeriodicReset(method, int(period)) if period else method
    raise ValueError(f"Unknown method kind {kind!r} (have: source, bn_adapt, tent)")
