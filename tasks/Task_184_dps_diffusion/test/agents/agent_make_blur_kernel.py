import matplotlib

matplotlib.use('Agg')

import torch

def make_blur_kernel(ksize: int, sigma: float) -> torch.Tensor:
    """Create a 2-D Gaussian blur kernel."""
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = g.outer(g)
    kernel /= kernel.sum()
    return kernel
