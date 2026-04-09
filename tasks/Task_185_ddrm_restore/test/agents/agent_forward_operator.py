import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.ndimage import gaussian_filter, zoom

def forward_operator(
    img_hr,
    scale_factor=4,
    aa_sigma=1.0,
    noise_std=0.05,
    seed=None
):
    """
    Apply the forward degradation model: y = A(x) + noise.
    
    The degradation consists of:
      1. Gaussian blur (anti-aliasing)
      2. Block-average downsampling
      3. Additive Gaussian noise
    
    Parameters
    ----------
    img_hr : ndarray, shape (H, W)
        High-resolution input image.
    scale_factor : int
        Downsampling factor.
    aa_sigma : float
        Sigma for Gaussian anti-aliasing blur.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int or None
        Random seed for noise generation.
        
    Returns
    -------
    lr_noisy : ndarray, shape (H//scale_factor, W//scale_factor)
        Noisy low-resolution observation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"[Forward] Blur(sigma={aa_sigma}) + {scale_factor}x block-avg "
          f"downsample + Noise(sigma={noise_std})")
    
    # Apply Gaussian blur (anti-aliasing filter)
    blurred = gaussian_filter(img_hr, sigma=aa_sigma)
    
    # Downsample by block averaging
    h, w = blurred.shape
    lr = blurred.reshape(h // scale_factor, scale_factor, 
                         w // scale_factor, scale_factor).mean(axis=(1, 3))
    
    # Add noise
    noise = np.random.randn(*lr.shape) * noise_std
    lr_noisy = np.clip(lr + noise, 0, 1)
    
    print(f"[Forward] Output shape: {lr_noisy.shape}, "
          f"range: [{lr_noisy.min():.4f}, {lr_noisy.max():.4f}]")
    return lr_noisy
