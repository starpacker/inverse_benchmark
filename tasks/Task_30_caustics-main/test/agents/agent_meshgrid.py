import torch

from typing import Optional, Union, Tuple, Literal, Annotated

from torch import Tensor

def meshgrid(pixelscale, nx, ny=None, device=None, dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    if ny is None:
        ny = nx
    xs = torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2
    ys = torch.linspace(-1, 1, ny, device=device, dtype=dtype) * pixelscale * (ny - 1) / 2
    return torch.meshgrid([xs, ys], indexing="xy")
