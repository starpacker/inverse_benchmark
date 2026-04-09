import matplotlib

matplotlib.use('Agg')

import torch

import torch.nn.functional as F

def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to image tensor (B,1,H,W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.view(1, 1, *kernel.shape), padding=pad)
