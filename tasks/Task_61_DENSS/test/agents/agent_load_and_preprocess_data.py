import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

from scipy.ndimage import gaussian_filter

def load_and_preprocess_data(n_grid, voxel_size, q_max, n_q, noise_pct, seed):
    """
    Generate synthetic SAXS data from a ground truth 3D electron density.

    Parameters
    ----------
    n_grid : int
        3D grid size.
    voxel_size : float
        Voxel size in Angstroms.
    q_max : float
        Maximum q value in inverse Angstroms.
    n_q : int
        Number of q bins.
    noise_pct : float
        Noise percentage to add to I(q).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'q': q values array
        - 'I_noisy': noisy scattering intensity
        - 'I_clean': clean scattering intensity
        - 'density_gt': ground truth 3D electron density
        - 'voxel_size': voxel size
        - 'q_max': maximum q
        - 'n_q': number of q bins
    """
    print("[DATA] Creating synthetic 3D electron density ...")
    
    # Create ground truth density (two-domain protein-like structure)
    rng = np.random.default_rng(seed)
    density_gt = np.zeros((n_grid, n_grid, n_grid))
    c = n_grid // 2

    # Main body (ellipsoidal)
    z, y, x = np.mgrid[:n_grid, :n_grid, :n_grid] - c
    ellipsoid = (x / 5) ** 2 + (y / 4) ** 2 + (z / 6) ** 2
    density_gt[ellipsoid < 1] = 1.0

    # Secondary domain (smaller sphere offset)
    sphere2 = (x - 4) ** 2 + (y + 3) ** 2 + (z - 2) ** 2
    density_gt[sphere2 < 9] = 0.8

    # Smooth with Gaussian
    density_gt = gaussian_filter(density_gt, sigma=1.0)

    # Normalise
    density_gt = density_gt / density_gt.max()

    print(f"[DATA] Density shape: {density_gt.shape}, "
          f"range [{density_gt.min():.3f}, {density_gt.max():.3f}]")

    print("[DATA] Computing SAXS intensity profile ...")
    q, I_clean = forward_operator(density_gt, voxel_size, n_q, q_max)

    # Add noise
    rng = np.random.default_rng(seed)
    I_noisy = I_clean * (1 + noise_pct * rng.standard_normal(n_q))
    I_noisy = np.maximum(I_noisy, 0)

    print(f"[DATA] I(q) range: [{I_clean.min():.3e}, {I_clean.max():.3e}]")

    return {
        'q': q,
        'I_noisy': I_noisy,
        'I_clean': I_clean,
        'density_gt': density_gt,
        'voxel_size': voxel_size,
        'q_max': q_max,
        'n_q': n_q
    }

def forward_operator(density, voxel_size, n_q, q_max):
    """
    Compute 1D SAXS profile I(q) from 3D electron density.

    I(q) = spherical_average( |FFT{ρ(r)}|² )

    Parameters
    ----------
    density : ndarray
        3D electron density array.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value in inverse Angstroms.

    Returns
    -------
    q_bins : ndarray
        q values in inverse Angstroms.
    I_q : ndarray
        Scattering intensity (normalized).
    """
    N = density.shape[0]

    # 3D FFT
    F = fftshift(fftn(ifftshift(density)))
    I_3d = np.abs(F) ** 2

    # q-grid
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Radial average (spherical shells)
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    I_q = np.zeros(n_q)

    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        if mask.sum() > 0:
            I_q[i] = np.mean(I_3d[mask])

    # Normalise
    if I_q.max() > 0:
        I_q = I_q / I_q.max()

    return q_bins, I_q
