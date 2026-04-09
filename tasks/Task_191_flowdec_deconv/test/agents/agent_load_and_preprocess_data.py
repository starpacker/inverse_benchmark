import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(vol_shape=(32, 128, 128), n_blobs=40, 
                              sigma_z=2.5, sigma_xy=1.2, seed=42):
    """
    Generate synthetic 3D fluorescence volume and PSF.
    
    Parameters
    ----------
    vol_shape : tuple
        (nz, ny, nx) volume dimensions.
    n_blobs : int
        Number of fluorescent blobs.
    sigma_z : float
        Standard deviation along z-axis (axial blur).
    sigma_xy : float
        Standard deviation along y and x axes (lateral blur).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    gt_volume : np.ndarray
        Ground truth 3D fluorescence volume.
    psf : np.ndarray
        3D point spread function.
    """
    # Generate synthetic 3D fluorescence volume
    rng = np.random.RandomState(seed)
    nz, ny, nx = vol_shape
    volume = np.zeros(vol_shape, dtype=np.float64)

    for _ in range(n_blobs):
        # Random center within the volume (with margin)
        cz = rng.randint(2, nz - 2)
        cy = rng.randint(10, ny - 10)
        cx = rng.randint(10, nx - 10)
        # Random radius in z and xy
        rz = rng.uniform(1.0, 2.5)
        rxy = rng.uniform(2.0, 5.0)
        intensity = rng.uniform(0.5, 1.0)

        # Create coordinate grids relative to blob center
        z_start = max(0, int(cz - 4*rz))
        z_end = min(nz, int(cz + 4*rz) + 1)
        y_start = max(0, int(cy - 4*rxy))
        y_end = min(ny, int(cy + 4*rxy) + 1)
        x_start = max(0, int(cx - 4*rxy))
        x_end = min(nx, int(cx + 4*rxy) + 1)
        
        zz, yy, xx = np.ogrid[z_start:z_end, y_start:y_end, x_start:x_end]
        blob = intensity * np.exp(
            -((zz - cz)**2 / (2 * rz**2)
              + (yy - cy)**2 / (2 * rxy**2)
              + (xx - cx)**2 / (2 * rxy**2))
        )
        volume[z_start:z_end, y_start:y_end, x_start:x_end] += blob

    # Add low-level background fluorescence
    volume += 0.02
    # Clip and normalize to [0, 1]
    volume = np.clip(volume, 0, None)
    volume /= volume.max()
    gt_volume = volume

    # Generate 3D PSF
    cz_psf, cy_psf, cx_psf = nz // 2, ny // 2, nx // 2
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    psf = np.exp(
        -((zz - cz_psf)**2 / (2 * sigma_z**2)
          + (yy - cy_psf)**2 / (2 * sigma_xy**2)
          + (xx - cx_psf)**2 / (2 * sigma_xy**2))
    )
    psf /= psf.sum()

    return gt_volume, psf
