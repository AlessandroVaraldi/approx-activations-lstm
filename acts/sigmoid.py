# acts/sigmoid.py
from __future__ import annotations

import numpy as np
import torch

# LUT parameters (be consistent across sigmoid/tanh)
_LUT_SIZE: int = 1024
_XMIN: float = -8.0
_XMAX: float = 8.0
_SCALE_BITS: int = 30
_SCALE: int = 1 << _SCALE_BITS  # Q30


def _build_sigmoid_lut() -> torch.Tensor:
    """
    Builds a _LUT_SIZE-entry sigmoid LUT in int32, storing y in Q30 (0..SCALE).
    x is uniformly sampled in [_XMIN, _XMAX].
    """
    xs = np.linspace(_XMIN, _XMAX, _LUT_SIZE, dtype=np.float64)

    # Numerically stable sigmoid in float64
    # sigmoid(x) = 1/(1+exp(-x))
    ys = 1.0 / (1.0 + np.exp(-xs))

    # Quantize to Q30 with round-to-nearest
    q = np.rint(ys * _SCALE).astype(np.int64)

    # Clamp to [0, SCALE] to be safe (avoid any accidental tiny overshoot)
    q = np.clip(q, 0, _SCALE).astype(np.int32)

    return torch.from_numpy(q)


# int32 LUT (size _LUT_SIZE) with sigmoid(x) in Q30, x uniformly sampled in [_XMIN, _XMAX]
_SIGMOID_LUT_I32: torch.Tensor = _build_sigmoid_lut()


def _x_to_index(x: torch.Tensor) -> torch.Tensor:
    """
    Maps x in float to LUT indices [0..255] using rounding.
    Saturates outside [_XMIN, _XMAX].
    """
    # clamp first for exact saturation behavior
    x_clamped = torch.clamp(x, _XMIN, _XMAX)

    # normalize to [0, 1]
    t = (x_clamped - _XMIN) / (_XMAX - _XMIN)

    # map to [0, LUT_SIZE-1] with round-to-nearest
    idx = torch.round(t * (_LUT_SIZE - 1)).to(torch.int64)

    # safety clamp
    idx = torch.clamp(idx, 0, _LUT_SIZE - 1)
    return idx


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    LUT-based sigmoid approximation.
    - Input: float tensor
    - Output: float tensor
    - LUT stored in int32 Q30
    Note: This function uses indexing; autograd backward will NOT be meaningful.
    Use STE or custom autograd if training requires gradients.
    """
    if not torch.is_tensor(x):
        raise TypeError("acts.sigmoid expects a torch.Tensor")

    # Ensure LUT on same device
    lut = _SIGMOID_LUT_I32.to(device=x.device)

    idx = _x_to_index(x)
    y_q30 = lut[idx]  # int32 gathered, broadcasted to x shape
    y = y_q30.to(torch.float32) / float(_SCALE)
    return y
