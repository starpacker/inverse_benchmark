import numpy as np

import matplotlib

matplotlib.use('Agg')

def _shepp_logan_ellipses_contrast(contrast='T1'):
    """
    Modified Shepp-Logan ellipses with tissue-dependent intensities.
    Each ellipse: (intensity, a, b, x0, y0, theta_deg)
    Tissues: outer skull, brain WM, GM regions, CSF ventricles, lesions.
    """
    if contrast == 'T1':
        return [
            (0.80, 0.6900, 0.9200, 0.0000, 0.0000, 0),
            (-0.60, 0.6624, 0.8740, 0.0000, -0.0184, 0),
            (0.50, 0.1100, 0.3100, 0.2200, 0.0000, -18),
            (0.50, 0.1600, 0.4100, -0.2200, 0.0000, 18),
            (0.30, 0.2100, 0.2500, 0.0000, 0.3500, 0),
            (0.25, 0.0460, 0.0460, 0.0000, 0.1000, 0),
            (0.25, 0.0460, 0.0460, 0.0000, -0.1000, 0),
            (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0),
            (0.10, 0.0230, 0.0230, 0.0000, -0.6050, 0),
            (0.15, 0.0230, 0.0460, 0.0600, -0.6050, 0),
        ]
    else:
        return [
            (0.70, 0.6900, 0.9200, 0.0000, 0.0000, 0),
            (-0.50, 0.6624, 0.8740, 0.0000, -0.0184, 0),
            (0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18),
            (0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18),
            (0.45, 0.2100, 0.2500, 0.0000, 0.3500, 0),
            (0.40, 0.0460, 0.0460, 0.0000, 0.1000, 0),
            (0.40, 0.0460, 0.0460, 0.0000, -0.1000, 0),
            (0.60, 0.0460, 0.0230, -0.0800, -0.6050, 0),
            (0.60, 0.0230, 0.0230, 0.0000, -0.6050, 0),
            (0.55, 0.0230, 0.0460, 0.0600, -0.6050, 0),
        ]

def generate_phantom(N, contrast='T1'):
    """Generate an NxN phantom image for a given contrast."""
    img = np.zeros((N, N), dtype=np.float64)
    ellipses = _shepp_logan_ellipses_contrast(contrast)

    y_coords, x_coords = np.mgrid[-1:1:N*1j, -1:1:N*1j]

    for (intensity, a, b, x0, y0, theta_deg) in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        xr = cos_t * (x_coords - x0) + sin_t * (y_coords - y0)
        yr = -sin_t * (x_coords - x0) + cos_t * (y_coords - y0)

        mask = (xr / a)**2 + (yr / b)**2 <= 1.0
        img[mask] += intensity

    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img

def create_cartesian_mask(N, acceleration, acs_fraction=0.08, seed=42):
    """
    Create a 1D Cartesian undersampling mask (same for all columns).
    Keeps center ACS lines and randomly selects remaining lines.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(N, dtype=bool)

    acs_lines = int(N * acs_fraction)
    center = N // 2
    acs_start = center - acs_lines // 2
    acs_end = acs_start + acs_lines
    mask[acs_start:acs_end] = True

    total_lines = N // acceleration
    remaining = max(0, total_lines - acs_lines)

    non_acs_indices = np.where(~mask)[0]
    if remaining > 0 and len(non_acs_indices) > 0:
        chosen = rng.choice(non_acs_indices, size=min(remaining, len(non_acs_indices)), replace=False)
        mask[chosen] = True

    mask_2d = np.zeros((N, N), dtype=bool)
    for i in range(N):
        if mask[i]:
            mask_2d[i, :] = True

    return mask_2d

def load_and_preprocess_data(N, t1_acceleration, t2_acceleration, acs_fraction=0.08, seed_t1=42, seed_t2=123):
    """
    Generate multi-contrast phantoms and create undersampling masks.
    
    Parameters:
    -----------
    N : int
        Image size (NxN)
    t1_acceleration : int
        Acceleration factor for T1 contrast
    t2_acceleration : int
        Acceleration factor for T2 contrast
    acs_fraction : float
        Fraction of center lines to keep as ACS
    seed_t1, seed_t2 : int
        Random seeds for mask generation
    
    Returns:
    --------
    data_dict : dict
        Contains ground truth images, masks, and undersampled k-space data
    """
    print("\n[1/4] Generating multi-contrast phantoms...")
    t1_gt = generate_phantom(N, contrast='T1')
    t2_gt = generate_phantom(N, contrast='T2')
    print(f"  T1 phantom: shape={t1_gt.shape}, range=[{t1_gt.min():.3f}, {t1_gt.max():.3f}]")
    print(f"  T2 phantom: shape={t2_gt.shape}, range=[{t2_gt.min():.3f}, {t2_gt.max():.3f}]")

    print("\n[2/4] Creating undersampling masks...")
    mask_t1 = create_cartesian_mask(N, acceleration=t1_acceleration, acs_fraction=acs_fraction, seed=seed_t1)
    mask_t2 = create_cartesian_mask(N, acceleration=t2_acceleration, acs_fraction=acs_fraction, seed=seed_t2)
    print(f"  T1 mask: {mask_t1.sum()/(N*N)*100:.1f}% sampled ({t1_acceleration}x acceleration)")
    print(f"  T2 mask: {mask_t2.sum()/(N*N)*100:.1f}% sampled ({t2_acceleration}x acceleration)")

    data_dict = {
        't1_gt': t1_gt,
        't2_gt': t2_gt,
        'mask_t1': mask_t1,
        'mask_t2': mask_t2,
        'N': N,
        't1_acceleration': t1_acceleration,
        't2_acceleration': t2_acceleration,
    }
    
    return data_dict
