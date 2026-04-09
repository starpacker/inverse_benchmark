import matplotlib

matplotlib.use('Agg')

import numpy as np

def load_and_preprocess_data(
    img_size=256,
    seed=42
):
    """
    Generate a synthetic ground-truth image for testing.
    
    Parameters
    ----------
    img_size : int
        Size of the square ground-truth image.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    gt_image : ndarray, shape (img_size, img_size)
        Synthetic ground-truth image normalized to [0, 1].
    """
    np.random.seed(seed)
    
    print("[GT] Generating synthetic ground truth image ...")
    size = img_size
    img = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]

    # Smooth gradient background
    img += 0.15 * (xx / size) + 0.1 * (yy / size)

    # Gaussian blob (soft circle)
    cx, cy = size * 0.35, size * 0.35
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    img += 0.25 * np.exp(-r**2 / (2 * (size * 0.12)**2))

    # Rectangle
    rect_mask = ((xx > size * 0.55) & (xx < size * 0.85) &
                 (yy > size * 0.15) & (yy < size * 0.45))
    img[rect_mask] += 0.5

    # Sinusoidal texture patch
    sin_mask = ((xx > size * 0.1) & (xx < size * 0.45) &
                (yy > size * 0.55) & (yy < size * 0.9))
    freq = 8.0 * np.pi / size
    img[sin_mask] += (0.2 * np.sin(freq * xx[sin_mask]) *
                      np.cos(freq * yy[sin_mask]) + 0.3)

    # Ellipse
    ecx, ecy = size * 0.7, size * 0.7
    a, b = size * 0.12, size * 0.08
    ellipse = ((xx - ecx) / a)**2 + ((yy - ecy) / b)**2
    img[ellipse < 1.0] += 0.6

    # Small bright dots (point sources)
    for dx, dy in [(0.2, 0.2), (0.8, 0.15), (0.85, 0.85), (0.15, 0.8)]:
        px, py = int(dx * size), int(dy * size)
        rr = np.sqrt((xx - px)**2 + (yy - py)**2)
        img += 0.4 * np.exp(-rr**2 / (2 * 3.0**2))

    # Diagonal stripe pattern
    stripe = 0.1 * np.sin(2 * np.pi * (xx + yy) / (size * 0.08))
    stripe_mask = (xx > size * 0.5) & (yy > size * 0.5) & (ellipse >= 1.0)
    img[stripe_mask] += stripe[stripe_mask]

    # Normalize to [0, 1]
    img = np.clip(img, 0, None)
    img = img / (img.max() + 1e-8)

    print(f"[GT] Image range: [{img.min():.4f}, {img.max():.4f}]")
    return img
