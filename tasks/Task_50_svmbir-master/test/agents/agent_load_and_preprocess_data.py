import numpy as np

import scipy.ndimage

try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

HAS_SKIMAGE = True

def _gen_ellipse(x_grid, y_grid, x0, y0, a, b, gray_level, theta=0):
    """Generates a single ellipse mask scaled by gray_level."""
    c = np.cos(theta)
    s = np.sin(theta)
    x_rot = (x_grid - x0) * c + (y_grid - y0) * s
    y_rot = -(x_grid - x0) * s + (y_grid - y0) * c
    mask = (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1.0
    return mask * gray_level

def _gen_shepp_logan(num_rows, num_cols):
    """Generates the Shepp-Logan phantom."""
    sl_paras = [
        {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.98},
        {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.02},
        {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.01}
    ]
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    x_grid, y_grid = np.meshgrid(axis_x, axis_y)
    image = np.zeros_like(x_grid)
    for el in sl_paras:
        image += _gen_ellipse(x_grid, y_grid, el['x0'], el['y0'], el['a'], el['b'], 
                              el['gray_level'], el['theta'] / 180.0 * np.pi)
    return image

def load_and_preprocess_data(image_size, num_views, noise_level=0.01):
    """
    Generates synthetic Shepp-Logan phantom data and creates a noisy sinogram.
    
    Returns:
        gt_image (np.array): Ground truth image.
        sinogram (np.array): Observed noisy sinogram.
        angles (np.array): Projection angles in radians.
    """
    # 1. Define geometry
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    
    # 2. Generate Ground Truth
    gt_image = _gen_shepp_logan(image_size, image_size)
    
    # 3. Simulate "Clean" Sinogram using Forward Operator
    # Note: We call forward_operator locally here to simulate data creation.
    clean_sino = forward_operator(gt_image, angles)
    
    # 4. Add Noise
    noise = np.random.normal(0, noise_level * np.max(clean_sino), clean_sino.shape)
    sinogram = clean_sino + noise
    
    return gt_image, sinogram, angles

def forward_operator(x, angles):
    """
    Computes the Radon Transform (Forward Projection).
    
    Args:
        x (np.array): Input image (N, N).
        angles (np.array): Projection angles in radians.
        
    Returns:
        y_pred (np.array): Sinogram (num_angles, num_detectors).
    """
    if HAS_SKIMAGE:
        theta_deg = np.degrees(angles)
        # skimage returns (num_det, num_angles), we want (num_angles, num_det)
        sino = radon(x, theta=theta_deg, circle=True)
        return sino.T
    else:
        num_angles = len(angles)
        num_rows, num_cols = x.shape
        num_det = num_cols 
        sinogram = np.zeros((num_angles, num_det), dtype=np.float32)
        
        for i, angle_rad in enumerate(angles):
            angle_deg = np.degrees(angle_rad)
            rotated_img = scipy.ndimage.rotate(x, angle_deg, reshape=False, order=1, mode='constant', cval=0.0)
            sinogram[i, :] = rotated_img.sum(axis=0)
        return sinogram
