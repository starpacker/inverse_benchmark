import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter

def create_phantom(N):
    """Create a synthetic 3D protein-like phantom volume."""
    vol = np.zeros((N, N, N), dtype=np.float64)
    center = N / 2.0
    z, y, x = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)

    r2 = (x - center)**2 + (y - center)**2 + (z - center)**2
    vol += 1.0 * np.exp(-r2 / (2 * (N * 0.15)**2))

    cx1, cy1, cz1 = center + N*0.18, center + N*0.12, center
    r2_1 = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    vol += 0.8 * np.exp(-r2_1 / (2 * (N * 0.10)**2))

    cx2, cy2, cz2 = center - N*0.15, center - N*0.10, center + N*0.12
    r2_2 = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    vol += 0.7 * np.exp(-r2_2 / (2 * (N * 0.08)**2))

    dist_cyl = np.sqrt((x - center - N*0.05)**2 + (y - center + N*0.15)**2)
    mask_cyl = (dist_cyl < N * 0.04) & (z > center - N*0.2) & (z < center + N*0.2)
    vol[mask_cyl] += 0.6

    for (dx, dy, dz) in [(0.1, 0.1, 0.15), (-0.12, 0.08, -0.1), (0.05, -0.15, 0.05)]:
        cx_s, cy_s, cz_s = center + N*dx, center + N*dy, center + N*dz
        r2_s = (x - cx_s)**2 + (y - cy_s)**2 + (z - cz_s)**2
        vol += 1.2 * np.exp(-r2_s / (2 * (N * 0.03)**2))

    vol = gaussian_filter(vol, sigma=1.0)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-10)
    return vol.astype(np.float32)
