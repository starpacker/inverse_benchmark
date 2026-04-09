import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def shepp_logan_phantom(N=256):
    """Generate modified Shepp-Logan phantom of size NxN."""
    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0.0, -0.605, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]
    img = np.zeros((N, N), dtype=np.float64)
    yc, xc = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    for val, a, b, x0, y0, ang in ellipses:
        th = np.radians(ang)
        ct, st = np.cos(th), np.sin(th)
        xr = ct * (xc - x0) + st * (yc - y0)
        yr = -st * (xc - x0) + ct * (yc - y0)
        img[(xr / a)**2 + (yr / b)**2 <= 1.0] += val
    return img

def create_cartesian_mask(N, acceleration=4, acs_fraction=0.08, seed=42):
    """Create 1D Cartesian undersampling mask with ACS lines."""
    rng = np.random.RandomState(seed)
    mask_1d = np.zeros(N, dtype=np.float64)
    
    acs_n = int(N * acs_fraction)
    c0 = N // 2 - acs_n // 2
    mask_1d[c0:c0 + acs_n] = 1.0
    
    target = N // acceleration
    needed = target - acs_n
    available = np.setdiff1d(np.arange(N), np.arange(c0, c0 + acs_n))
    if needed > 0:
        chosen = rng.choice(available, min(needed, len(available)), replace=False)
        mask_1d[chosen] = 1.0
    
    mask_2d = np.tile(mask_1d[:, None], (1, N))
    rate = mask_1d.sum() / N
    print(f"  Undersampling mask: {int(mask_1d.sum())}/{N} lines "
          f"({rate*100:.1f}%), ~{1/rate:.1f}x acceleration")
    return mask_2d

def load_and_preprocess_data(N=256, acceleration=4, acs_fraction=0.08, seed=42):
    """
    Load and preprocess data for MRI reconstruction.
    
    Generates:
      - Ground truth Shepp-Logan phantom
      - Undersampling mask
      - Undersampled k-space measurements
      - Zero-filled reconstruction (baseline)
    
    Returns:
        dict containing:
            'gt_image': Ground truth phantom image
            'mask': Undersampling mask
            'y_kspace': Undersampled k-space data
            'zero_filled': Zero-filled reconstruction
            'N': Image size
    """
    print("\n[1/3] Generating Shepp-Logan phantom...")
    gt_image = shepp_logan_phantom(N)
    print(f"  Phantom range: [{gt_image.min():.4f}, {gt_image.max():.4f}]")
    
    print("\n[2/3] Creating undersampling mask...")
    mask = create_cartesian_mask(N, acceleration=acceleration, 
                                  acs_fraction=acs_fraction, seed=seed)
    
    print("\n[3/3] Simulating undersampled k-space acquisition...")
    full_kspace = np.fft.fft2(gt_image, norm='ortho')
    y_kspace = mask * full_kspace
    
    zero_filled = np.real(np.fft.ifft2(y_kspace, norm='ortho'))
    
    return {
        'gt_image': gt_image,
        'mask': mask,
        'y_kspace': y_kspace,
        'zero_filled': zero_filled,
        'N': N
    }
