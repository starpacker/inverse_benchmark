import matplotlib

matplotlib.use('Agg')

import numpy as np

from skimage.transform import radon, iradon

def shepp_logan_phantom(size=128):
    """Generate Shepp-Logan phantom."""
    p = np.zeros((size, size), dtype=np.float64)
    els = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),
        (0.06, -0.605, 0.046, 0.023, 0, 0.01),
    ]
    c = size // 2
    for cy, cx, a, b, ang, I in els:
        cy_px, cx_px = c + cy * size / 2, c + cx * size / 2
        a_px, b_px = a * size / 2, b * size / 2
        ar = np.radians(ang)
        yy, xx = np.mgrid[:size, :size]
        ca, sa = np.cos(ar), np.sin(ar)
        xr = ca * (xx - cx_px) + sa * (yy - cy_px)
        yr = -sa * (xx - cx_px) + ca * (yy - cy_px)
        p[(xr / a_px)**2 + (yr / b_px)**2 <= 1.0] += I
    p = np.clip(p, 0, None)
    p /= p.max() + 1e-12
    return p

def load_and_preprocess_data(size=128, n_angles=75, noise_level=0.02, seed=42):
    """
    Generate phantom, compute sinogram with noise, and return all data needed
    for reconstruction.

    Returns
    -------
    data : dict with keys:
        'phantom'    : np.ndarray (size, size) ground truth
        'sinogram'   : np.ndarray noisy sinogram
        'theta'      : np.ndarray projection angles
        'size'       : int image size
        'n_angles'   : int number of angles
        'noise_level': float
    """
    phantom = shepp_logan_phantom(size)
    print(f"[1] Phantom {size}x{size}")

    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(phantom, theta=theta)
    rng = np.random.RandomState(seed)
    sino_n = sino + rng.normal(0, noise_level * sino.max(), sino.shape)
    print(f"[2] Sinogram: {n_angles} angles, {int(noise_level*100)}% noise, shape={sino_n.shape}")

    data = {
        'phantom': phantom,
        'sinogram': sino_n,
        'theta': theta,
        'size': size,
        'n_angles': n_angles,
        'noise_level': noise_level,
    }
    return data
