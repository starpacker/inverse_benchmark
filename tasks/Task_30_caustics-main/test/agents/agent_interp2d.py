import torch

from typing import Optional, Union, Tuple, Literal, Annotated

from torch import Tensor

def interp2d(im: Tensor, x: Tensor, y: Tensor, method: Literal["linear", "nearest"] = "linear", padding_mode: str = "zeros") -> Tensor:
    if im.ndim != 2:
        raise ValueError(f"im must be 2D (received {im.ndim}D tensor)")
    if padding_mode == "clamp":
        x = x.clamp(-1, 1)
        y = y.clamp(-1, 1)
    else:
        idxs_out_of_bounds = (y < -1) | (y > 1) | (x < -1) | (x > 1)

    h, w = im.shape
    x = 0.5 * ((x + 1) * w - 1)
    y = 0.5 * ((y + 1) * h - 1)

    if method == "nearest":
        result = im[y.round().long().clamp(0, h - 1), x.round().long().clamp(0, w - 1)]
    elif method == "linear":
        x0 = x.floor().long().clamp(0, w - 2)
        y0 = y.floor().long().clamp(0, h - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        fa = im[y0, x0]
        fb = im[y1, x0]
        fc = im[y0, x1]
        fd = im[y1, x1]
        dx1 = x1 - x
        dx0 = x - x0
        dy1 = y1 - y
        dy0 = y - y0
        result = fa * dx1 * dy1 + fb * dx1 * dy0 + fc * dx0 * dy1 + fd * dx0 * dy0
    
    if padding_mode == "zeros":
        result = torch.where(idxs_out_of_bounds, torch.zeros_like(result), result)
    return result
