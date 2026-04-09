import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def shepp_logan_phantom(n=128):
    """Generate a Shepp-Logan phantom of size n x n."""
    ellipses = [
        (1.0,   0.6900, 0.9200,  0.0000,  0.0000,   0),
        (-0.8,  0.6624, 0.8740,  0.0000, -0.0184,   0),
        (-0.2,  0.1100, 0.3100,  0.2200,  0.0000, -18),
        (-0.2,  0.1600, 0.4100, -0.2200,  0.0000,  18),
        (0.1,   0.2100, 0.2500,  0.0000,  0.3500,   0),
        (0.1,   0.0460, 0.0460,  0.0000,  0.1000,   0),
        (0.1,   0.0460, 0.0460,  0.0000, -0.1000,   0),
        (0.1,   0.0460, 0.0230, -0.0800, -0.6050,   0),
        (0.1,   0.0230, 0.0230,  0.0000, -0.6060,   0),
        (0.1,   0.0230, 0.0460,  0.0600, -0.6050,   0),
    ]

    phantom = np.zeros((n, n), dtype=np.float64)
    y_coords, x_coords = np.mgrid[-1:1:n*1j, -1:1:n*1j]

    for intensity, a, b, x0, y0, theta_deg in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = cos_t * (x_coords - x0) + sin_t * (y_coords - y0)
        y_rot = -sin_t * (x_coords - x0) + cos_t * (y_coords - y0)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
        phantom[mask] += intensity

    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    return phantom

def create_undersampling_mask(shape, acceleration=4, center_fraction=0.08, seed=42):
    """
    Create a random Cartesian undersampling mask.
    """
    rng = np.random.RandomState(seed)
    ny, nx = shape
    mask = np.zeros(shape, dtype=np.float64)

    num_center = int(center_fraction * ny)
    center_start = (ny - num_center) // 2
    mask[center_start:center_start + num_center, :] = 1.0

    num_total_lines = ny // acceleration
    num_random_lines = max(num_total_lines - num_center, 0)

    available = list(set(range(ny)) - set(range(center_start, center_start + num_center)))
    if num_random_lines > 0 and len(available) > 0:
        chosen = rng.choice(available, size=min(num_random_lines, len(available)), replace=False)
        for idx in chosen:
            mask[idx, :] = 1.0

    return mask

def load_and_preprocess_data(image_size=128, acceleration=4, center_fraction=0.08, seed=42):
    """
    Generate Shepp-Logan phantom and create undersampling mask.
    
    Args:
        image_size: size of the phantom image (N x N)
        acceleration: undersampling factor
        center_fraction: fraction of center k-space lines to always sample
        seed: random seed for reproducibility
    
    Returns:
        gt_image: ground truth Shepp-Logan phantom (numpy array)
        mask: undersampling mask (numpy array)
        params: dictionary with parameters
    """
    print("\n[load_and_preprocess_data] Generating Shepp-Logan phantom...")
    gt_image = shepp_logan_phantom(image_size)
    print(f"  Phantom range: [{gt_image.min():.4f}, {gt_image.max():.4f}]")
    
    print("[load_and_preprocess_data] Creating undersampling mask...")
    mask = create_undersampling_mask(gt_image.shape, acceleration=acceleration,
                                     center_fraction=center_fraction, seed=seed)
    sampling_ratio = mask.sum() / mask.size
    print(f"  Sampling ratio: {sampling_ratio:.2%} (target ~{100/acceleration:.0f}%)")
    
    params = {
        'image_size': image_size,
        'acceleration': acceleration,
        'center_fraction': center_fraction,
        'seed': seed,
        'sampling_ratio': float(sampling_ratio)
    }
    
    return gt_image, mask, params
