import matplotlib

matplotlib.use('Agg')

import torch

import torch.nn.functional as F

def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to image tensor (B,1,H,W)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.view(1, 1, *kernel.shape), padding=pad)

def forward_operator(x: torch.Tensor, blur_kernel: torch.Tensor,
                     noise_std: float) -> torch.Tensor:
    """
    Apply the forward degradation operator: y = blur(x) + N(0, σ²).
    
    Args:
        x: Input image tensor (B, C, H, W)
        blur_kernel: Gaussian blur kernel
        noise_std: Standard deviation of additive Gaussian noise
        
    Returns:
        y: Degraded observation tensor (B, C, H, W)
    """
    print("\n[2/6] Applying forward operator (Gaussian blur + noise) ...")
    blurred = apply_blur(x, blur_kernel)
    noise = torch.randn_like(blurred) * noise_std
    y = blurred + noise
    
    y_np = y.squeeze().cpu().numpy()
    print(f"  Degraded range: [{y_np.min():.3f}, {y_np.max():.3f}]")
    
    return y
