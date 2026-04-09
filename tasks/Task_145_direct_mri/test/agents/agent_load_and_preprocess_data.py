import matplotlib

matplotlib.use('Agg')

import numpy as np

def shepp_logan_phantom(N=128):
    """Generate a Shepp-Logan phantom of size NxN."""
    ellipses = [
        (2.0, 0.6900, 0.9200, 0.0000, 0.0000, 0),
        (-0.98, 0.6624, 0.8740, 0.0000, -0.0184, 0),
        (-0.02, 0.1100, 0.3100, 0.2200, 0.0000, -18),
        (-0.02, 0.1600, 0.4100, -0.2200, 0.0000, 18),
        (0.01, 0.2100, 0.2500, 0.0000, 0.3500, 0),
        (0.01, 0.0460, 0.0460, 0.0000, 0.1000, 0),
        (0.01, 0.0460, 0.0460, 0.0000, -0.1000, 0),
        (0.01, 0.0460, 0.0230, -0.0800, -0.6050, 0),
        (0.01, 0.0230, 0.0230, 0.0000, -0.6060, 0),
        (0.01, 0.0230, 0.0460, 0.0600, -0.6050, 0),
    ]

    img = np.zeros((N, N), dtype=np.float64)
    ygrid, xgrid = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]

    for intensity, a, b, x0, y0, theta_deg in ellipses:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = cos_t * (xgrid - x0) + sin_t * (ygrid - y0)
        yr = -sin_t * (xgrid - x0) + cos_t * (ygrid - y0)
        region = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        img[region] += intensity

    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img

def fft2c(img):
    """Centered 2D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    """Centered 2D IFFT: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def create_undersampling_mask(N, acceleration=4, acs_lines=16, seed=42):
    """Create a random Cartesian undersampling mask (row-based)."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((N, N), dtype=np.float64)

    center = N // 2
    acs_start = center - acs_lines // 2
    acs_end = center + acs_lines // 2
    mask[acs_start:acs_end, :] = 1.0

    total_lines_needed = N // acceleration
    acs_count = acs_end - acs_start
    remaining_needed = max(0, total_lines_needed - acs_count)

    available = list(set(range(N)) - set(range(acs_start, acs_end)))
    chosen = rng.choice(available, size=min(remaining_needed, len(available)), replace=False)
    for idx in chosen:
        mask[idx, :] = 1.0

    return mask

def load_and_preprocess_data(N=128, acceleration=4, acs_lines=16, seed=42):
    """
    Generate synthetic MRI data: Shepp-Logan phantom, undersampling mask,
    and undersampled k-space.
    
    Returns:
        gt_image: Ground truth image (NxN)
        mask: Undersampling mask (NxN)
        kspace_under: Undersampled k-space data (NxN complex)
        gt_norm: Normalized ground truth
        zf_recon: Zero-filled reconstruction
        zf_norm: Normalized zero-filled reconstruction
    """
    # Generate phantom
    gt_image = shepp_logan_phantom(N)
    
    # Create undersampling mask
    mask = create_undersampling_mask(N, acceleration=acceleration, acs_lines=acs_lines, seed=seed)
    
    # Forward model: get full k-space and undersample
    kspace_full = fft2c(gt_image)
    kspace_under = kspace_full * mask
    
    # Normalize ground truth
    gt_norm = gt_image / (gt_image.max() + 1e-12)
    
    # Zero-filled reconstruction (baseline)
    zf_recon = np.abs(ifft2c(kspace_under))
    zf_norm = zf_recon / (zf_recon.max() + 1e-12)
    
    return {
        'gt_image': gt_image,
        'gt_norm': gt_norm,
        'mask': mask,
        'kspace_under': kspace_under,
        'zf_recon': zf_recon,
        'zf_norm': zf_norm,
        'N': N,
        'acceleration': acceleration
    }
