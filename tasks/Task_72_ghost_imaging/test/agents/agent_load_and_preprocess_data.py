import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter

def load_and_preprocess_data(img_size, compression_ratio, noise_snr_db, seed):
    """
    Generate test image, speckle patterns, and bucket measurements.
    
    Returns:
        dict containing:
            - img_gt: ground truth image (2D)
            - x_gt: vectorized ground truth
            - Phi: measurement matrix (speckle patterns)
            - b_clean: clean bucket measurements
            - b_noisy: noisy bucket measurements
            - config: configuration parameters
    """
    rng = np.random.default_rng(seed)
    n_pixels = img_size ** 2
    n_measurements = int(compression_ratio * n_pixels)
    
    # Generate test image with geometric features
    img_gt = np.zeros((img_size, img_size))
    cx, cy = img_size // 2, img_size // 2
    
    # Circle
    Y, X = np.mgrid[:img_size, :img_size]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    img_gt[r < img_size // 4] = 0.8
    
    # Square
    sq_s = img_size // 8
    img_gt[cx-sq_s:cx+sq_s, img_size//4-sq_s:img_size//4+sq_s] = 0.6
    
    # Triangle
    for i in range(img_size // 6):
        j_start = 3 * img_size // 4 - i
        j_end = 3 * img_size // 4 + i
        row = cy - img_size // 6 + i
        if 0 <= row < img_size and 0 <= j_start and j_end < img_size:
            img_gt[row, j_start:j_end] = 0.7
    
    # Small bright features
    img_gt[img_size // 6, img_size // 6] = 1.0
    img_gt[5 * img_size // 6, img_size // 6] = 1.0
    img_gt[img_size // 6, 5 * img_size // 6] = 1.0
    img_gt[5 * img_size // 6, 5 * img_size // 6] = 1.0
    
    # Smooth and normalize
    img_gt = gaussian_filter(img_gt, sigma=1)
    img_gt = img_gt / max(img_gt.max(), 1e-12)
    
    x_gt = img_gt.ravel()
    
    # Generate speckle patterns (Gaussian random for better RIP properties)
    Phi = rng.standard_normal((n_measurements, n_pixels)) / np.sqrt(n_measurements)
    
    # Forward measurement with noise
    b_clean = Phi @ x_gt
    sig_power = np.mean(b_clean**2)
    noise_power = sig_power / (10**(noise_snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(len(b_clean))
    b_noisy = b_clean + noise
    
    config = {
        'img_size': img_size,
        'n_pixels': n_pixels,
        'n_measurements': n_measurements,
        'compression_ratio': compression_ratio,
        'noise_snr_db': noise_snr_db,
        'seed': seed
    }
    
    return {
        'img_gt': img_gt,
        'x_gt': x_gt,
        'Phi': Phi,
        'b_clean': b_clean,
        'b_noisy': b_noisy,
        'config': config
    }
