# acts/tanh.py
from __future__ import annotations

import numpy as np
import torch

_LUT_SIZE: int = 1024
_XMIN: float = -8.0
_XMAX: float = 8.0
_SCALE_BITS: int = 30
_SCALE: int = 1 << _SCALE_BITS  # Q30


def _build_tanh_lut() -> torch.Tensor:
    """
    Builds a _LUT_SIZE-entry tanh LUT in int32, storing y in Q30 (-SCALE..SCALE).
    """
    xs = np.linspace(_XMIN, _XMAX, _LUT_SIZE, dtype=np.float64)
    ys = np.tanh(xs)

    q = np.rint(ys * _SCALE).astype(np.int64)

    # Clamp to [-SCALE, SCALE]
    q = np.clip(q, -_SCALE, _SCALE).astype(np.int32)

    return torch.from_numpy(q)


_TANH_LUT_I32: torch.Tensor = _build_tanh_lut()


def _x_to_index(x: torch.Tensor) -> torch.Tensor:
    x_clamped = torch.clamp(x, _XMIN, _XMAX)
    t = (x_clamped - _XMIN) / (_XMAX - _XMIN)
    idx = torch.round(t * (_LUT_SIZE - 1)).to(torch.int64)
    idx = torch.clamp(idx, 0, _LUT_SIZE - 1)
    return idx


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    LUT-based tanh approximation.
    - Input: float tensor
    - Output: float tensor
    - LUT stored in int32 Q30
    Note: autograd backward will NOT be meaningful; use STE if needed.
    """
    if not torch.is_tensor(x):
        raise TypeError("acts.tanh expects a torch.Tensor")

    lut = _TANH_LUT_I32.to(device=x.device)

    idx = _x_to_index(x)
    y_q30 = lut[idx]
    y = y_q30.to(torch.float32) / float(_SCALE)
    return y
