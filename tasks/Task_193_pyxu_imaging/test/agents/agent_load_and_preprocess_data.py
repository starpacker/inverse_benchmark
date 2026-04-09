import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

import pyxu.operator as pxo

def create_gaussian_kernel(sigma, size=None):
    """Create a 2D Gaussian convolution kernel."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def load_and_preprocess_data(img_size, blur_sigma, noise_level, seed):
    """
    Create synthetic test image and generate blurred noisy observation.
    
    Parameters:
    -----------
    img_size : int
        Size of the square image (img_size x img_size)
    blur_sigma : float
        Standard deviation of Gaussian blur kernel
    noise_level : float
        Standard deviation of additive Gaussian noise
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    x_true : np.ndarray
        Ground truth image (2D)
    y_observed : np.ndarray
        Blurred and noisy observation (1D flattened)
    kernel : np.ndarray
        Gaussian blur kernel (2D)
    """
    # Create synthetic test image
    rng = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size), dtype=np.float64)

    # Background gradient
    y_coords, x_coords = np.mgrid[0:img_size, 0:img_size] / float(img_size)
    img += 0.1 * y_coords

    # Rectangle
    img[20:50, 30:80] = 0.7

    # Circle
    cy, cx, r = 80, 40, 20
    yy, xx = np.ogrid[:img_size, :img_size]
    mask_circle = (yy - cy)**2 + (xx - cx)**2 <= r**2
    img[mask_circle] = 0.9

    # Small bright square
    img[60:75, 85:100] = 1.0

    # Triangle
    for row in range(30, 60):
        col_start = 85 + (row - 30)
        col_end = 115 - (row - 30)
        if col_start < col_end and col_end <= img_size:
            img[row, col_start:col_end] = 0.5

    # Diagonal stripe
    for i in range(img_size):
        j_start = max(0, i - 3)
        j_end = min(img_size, i + 3)
        if 10 <= i <= 110:
            img[i, j_start:j_end] = np.maximum(img[i, j_start:j_end], 0.4)

    # Small dots
    dot_positions = [(15, 15), (15, 110), (110, 15), (110, 110), (64, 64)]
    for dy, dx in dot_positions:
        if 0 <= dy < img_size and 0 <= dx < img_size:
            img[max(0, dy-2):min(img_size, dy+3), max(0, dx-2):min(img_size, dx+3)] = 0.85

    x_true = np.clip(img, 0, 1)
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(blur_sigma)
    
    # Build forward operator for generating observation
    H = pxo.Convolve(
        arg_shape=x_true.shape,
        kernel=kernel,
        center=(kernel.shape[0]//2, kernel.shape[1]//2),
        mode="constant",
    )
    
    # Generate observation: y = H(x) + noise
    x_true_flat = x_true.ravel()
    rng_noise = np.random.RandomState(seed + 1)
    y_clean = H(x_true_flat)
    noise = rng_noise.normal(0, noise_level, y_clean.shape)
    y_observed = y_clean + noise
    
    print(f"  Image shape: {x_true.shape}, range: [{x_true.min():.3f}, {x_true.max():.3f}]")
    print(f"  Kernel shape: {kernel.shape}, sigma={blur_sigma}")
    print(f"  Observation shape: {y_observed.shape}")
    
    return x_true, y_observed, kernel
