"""uint8 NHWC numpy batch -> float [0,1] NCHW tensor, with an explicit memory layout.

`permute(0,3,1,2)` of an NHWC array is a VIEW with channels_last strides, and `.float()` preserves strides, so the
naive conversion returns a channels_last-strided tensor and the whole network then runs in channels_last (cuDNN
NHWC kernels). `layout="nchw"` forces a contiguous NCHW tensor. Values are identical; only memory layout differs.
"""
from __future__ import annotations

import numpy as np
import torch


def to_tensor(x_uint8: np.ndarray, device: torch.device, layout: str = "as_is") -> torch.Tensor:
    t = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2)
    if layout == "nchw":
        t = t.contiguous()                      # contiguous_format = standard NCHW strides
    elif layout != "as_is":
        raise ValueError(f"layout must be 'as_is' or 'nchw', got {layout!r}")
    return t.float().div_(255.0)
