import matplotlib

matplotlib.use("Agg")

import numpy as np

def _line_pixel_lengths(r_arr, z_arr, p0, p1, dr, dz):
    """
    Compute the intersection lengths of the line segment p0→p1 with each
    pixel on the (r_arr, z_arr) grid using Siddon's algorithm (simplified).

    Returns (row_indices, values) for one LOS.
    """
    nr, nz = len(r_arr), len(z_arr)
    r0, z0 = p0
    r1, z1 = p1

    # Total line length
    total_len = np.hypot(r1 - r0, z1 - z0)
    if total_len < 1e-12:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Parametric: P(t) = p0 + t*(p1-p0), t in [0,1]
    n_samples = max(int(total_len / (min(dr, dz) * 0.25)), 500)
    t = np.linspace(0, 1, n_samples)
    r_pts = r0 + t * (r1 - r0)
    z_pts = z0 + t * (z1 - z0)

    # Map to grid indices
    ir = np.floor((r_pts - r_arr[0]) / dr).astype(int)
    iz = np.floor((z_pts - z_arr[0]) / dz).astype(int)

    # Mask valid
    valid = (ir >= 0) & (ir < nr) & (iz >= 0) & (iz < nz)
    ir = ir[valid]
    iz = iz[valid]

    if len(ir) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    flat = ir * nz + iz
    dl = total_len / n_samples  # approximate step length

    # Accumulate
    unique_idx, inverse = np.unique(flat, return_inverse=True)
    weights = np.bincount(inverse).astype(float) * dl

    return unique_idx, weights
