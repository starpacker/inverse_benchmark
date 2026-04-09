import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy import sparse

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

def build_geometry_matrix(r_arr, z_arr, n_detectors, n_los_per_det, r_min, r_max, z_min, z_max):
    """
    Build the sparse geometry (line-integral) matrix L of shape
    (n_los_total, NR*NZ).

    Detector fans are placed around the vessel at different poloidal angles.
    """
    dr = r_arr[1] - r_arr[0]
    dz = z_arr[1] - z_arr[0]
    nr, nz = len(r_arr), len(z_arr)
    n_pix = nr * nz

    # Vessel centre for detector placement
    R_center = 0.5 * (r_min + r_max)
    Z_center = 0.5 * (z_min + z_max)
    vessel_radius = 1.0  # approximate distance from centre to wall

    # Detector angular positions (poloidal angle around vessel cross-section)
    det_angles = np.linspace(0, 2 * np.pi, n_detectors, endpoint=False)

    rows, cols, vals = [], [], []
    los_idx = 0

    for da in det_angles:
        # Detector position on vessel wall
        det_r = R_center + vessel_radius * np.cos(da)
        det_z = Z_center + vessel_radius * np.sin(da)

        # Fan of LOS aiming through the plasma
        # Compute angular span that covers the plasma region
        fan_half = np.deg2rad(35)
        fan_angles = np.linspace(da + np.pi - fan_half,
                                 da + np.pi + fan_half,
                                 n_los_per_det)

        for fa in fan_angles:
            # End-point on opposite side of vessel
            end_r = det_r + 2.5 * vessel_radius * np.cos(fa)
            end_z = det_z + 2.5 * vessel_radius * np.sin(fa)

            idx, wts = _line_pixel_lengths(
                r_arr, z_arr, (det_r, det_z), (end_r, end_z), dr, dz
            )
            if len(idx) > 0:
                rows.extend([los_idx] * len(idx))
                cols.extend(idx.tolist())
                vals.extend(wts.tolist())
            los_idx += 1

    n_los = los_idx
    L = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_los, n_pix), dtype=np.float64
    )
    return L, n_los
