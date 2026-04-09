import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import map_coordinates

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_speckle_image(height, width, n_speckles=15000,
                           speckle_sigma=2.5, seed=42):
    """Generate a dense synthetic speckle pattern for DIC."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.float64)

    ys = rng.uniform(0, height, n_speckles)
    xs = rng.uniform(0, width, n_speckles)
    intensities = rng.uniform(0.3, 1.0, n_speckles)

    r = int(4 * speckle_sigma) + 1
    for y0, x0, amp in zip(ys, xs, intensities):
        y_lo = max(0, int(y0) - r)
        y_hi = min(height, int(y0) + r + 1)
        x_lo = max(0, int(x0) - r)
        x_hi = min(width, int(x0) + r + 1)
        yy = np.arange(y_lo, y_hi)[:, None]
        xx = np.arange(x_lo, x_hi)[None, :]
        gauss = amp * np.exp(-((yy - y0)**2 + (xx - x0)**2) /
                              (2 * speckle_sigma**2))
        img[y_lo:y_hi, x_lo:x_hi] += gauss

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

def generate_displacement_fields(height, width, n_frames=10):
    """Generate smooth spatially-varying sinusoidal displacement fields."""
    Y, X = np.meshgrid(np.arange(height, dtype=np.float64),
                        np.arange(width, dtype=np.float64), indexing='ij')

    sigma_spatial = min(height, width) / 2.0
    envelope = np.exp(-((X - width / 2)**2 + (Y - height / 2)**2) /
                       (2 * sigma_spatial**2))

    dx_fields = np.zeros((n_frames, height, width))
    dy_fields = np.zeros((n_frames, height, width))

    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames
        amp_x = 2.5 * np.sin(phase)
        amp_y = 1.8 * np.cos(phase)
        dx_fields[t] = amp_x * envelope
        dy_fields[t] = amp_y * envelope

    return dx_fields, dy_fields

def load_and_preprocess_data(height, width, n_frames, noise_sigma=0.001):
    """
    Generate synthetic data for DIC displacement tracking.
    
    Returns:
        ref_image: Reference speckle image (height, width)
        images: Deformed image sequence (n_frames, height, width)
        dx_gt: Ground truth x-displacement fields (n_frames, height, width)
        dy_gt: Ground truth y-displacement fields (n_frames, height, width)
        params: Dictionary containing preprocessing parameters
    """
    # Generate reference speckle image
    ref_image = generate_speckle_image(height, width)
    
    # Generate ground truth displacement fields
    dx_gt, dy_gt = generate_displacement_fields(height, width, n_frames)
    
    # Generate deformed image sequence using forward operator internally
    rng = np.random.RandomState(123)
    images = []
    for t in range(n_frames):
        # Warp image by displacement field using backward mapping
        h, w = ref_image.shape
        rr, cc = np.meshgrid(np.arange(h, dtype=np.float64),
                             np.arange(w, dtype=np.float64), indexing='ij')
        warped = map_coordinates(ref_image, [rr - dy_gt[t], cc - dx_gt[t]],
                                 order=3, mode='reflect')
        noise = rng.normal(0, noise_sigma, warped.shape)
        warped = np.clip(warped + noise, 0, 1)
        images.append(warped)
    images = np.array(images)
    
    params = {
        'height': height,
        'width': width,
        'n_frames': n_frames,
        'noise_sigma': noise_sigma
    }
    
    return ref_image, images, dx_gt, dy_gt, params
