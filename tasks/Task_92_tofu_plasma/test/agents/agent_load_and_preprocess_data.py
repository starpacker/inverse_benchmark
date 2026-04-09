import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy import sparse

def make_grid(nr, nz, r_min, r_max, z_min, z_max):
    """Return 1-D R, Z arrays and the 2-D meshgrid."""
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")  # shape (NR, NZ)
    return r, z, RR, ZZ

def make_phantom(RR, ZZ):
    """
    Create a realistic tokamak-like emission phantom.
    Peaked profile centred at (R0, Z0) with an elliptical Gaussian shape
    plus a secondary weaker blob to add asymmetry.
    """
    R0, Z0 = 1.75, 0.0       # magnetic axis
    sigma_r, sigma_z = 0.30, 0.35

    # Main peaked profile
    eps = np.exp(-((RR - R0) ** 2 / (2 * sigma_r ** 2)
                   + (ZZ - Z0) ** 2 / (2 * sigma_z ** 2)))

    # Secondary blob (HFS accumulation)
    R1, Z1 = 1.45, 0.15
    sig1_r, sig1_z = 0.12, 0.10
    eps += 0.35 * np.exp(-((RR - R1) ** 2 / (2 * sig1_r ** 2)
                           + (ZZ - Z1) ** 2 / (2 * sig1_z ** 2)))

    # Clip outside last closed flux surface (rough ellipse)
    a_r, a_z = 0.60, 0.70
    mask = ((RR - R0) / a_r) ** 2 + ((ZZ - Z0) / a_z) ** 2 <= 1.0
    eps *= mask.astype(float)

    # Normalise to [0, 1]
    eps /= eps.max()
    return eps

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

def load_and_preprocess_data(nr, nz, r_min, r_max, z_min, z_max, 
                              n_detectors, n_los_per_det, noise_level, rng_seed):
    """
    Load and preprocess data for plasma tomography reconstruction.
    
    Creates the computational grid, generates the phantom emissivity,
    builds the geometry matrix, performs forward projection, and adds noise.
    
    Parameters
    ----------
    nr : int
        Number of grid points in R direction
    nz : int
        Number of grid points in Z direction
    r_min, r_max : float
        Major radius range [m]
    z_min, z_max : float
        Vertical range [m]
    n_detectors : int
        Number of detector fans
    n_los_per_det : int
        Lines-of-sight per detector fan
    noise_level : float
        Relative Gaussian noise level on measurements
    rng_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'r_arr': 1D array of R coordinates
        - 'z_arr': 1D array of Z coordinates
        - 'ground_truth': 2D ground truth emissivity array
        - 'geometry_matrix': sparse geometry matrix L
        - 'measurements_clean': clean line-integrated measurements
        - 'measurements_noisy': noisy measurements
        - 'n_los': number of lines of sight
        - 'nr': number of R grid points
        - 'nz': number of Z grid points
        - 'n_detectors': number of detector fans
        - 'n_los_per_det': LOS per detector
    """
    rng = np.random.default_rng(rng_seed)
    
    # Build grid and phantom
    print("[1/6] Building grid and phantom …")
    r_arr, z_arr, RR, ZZ = make_grid(nr, nz, r_min, r_max, z_min, z_max)
    gt_2d = make_phantom(RR, ZZ)
    gt_flat = gt_2d.ravel()
    
    # Build geometry matrix
    print("[2/6] Building geometry matrix (line-of-sight integrals) …")
    L, n_los = build_geometry_matrix(r_arr, z_arr, n_detectors, n_los_per_det, 
                                      r_min, r_max, z_min, z_max)
    print(f"       Geometry matrix: {L.shape[0]} LOS × {L.shape[1]} pixels, "
          f"nnz = {L.nnz}")
    
    # Forward projection with noise
    print("[3/6] Forward projection + noise …")
    y_clean = L @ gt_flat
    sigma_noise = noise_level * np.max(np.abs(y_clean))
    noise = rng.normal(0, sigma_noise, size=y_clean.shape)
    y_noisy = y_clean + noise
    print(f"       SNR ≈ {np.linalg.norm(y_clean) / np.linalg.norm(noise):.1f}")
    
    return {
        'r_arr': r_arr,
        'z_arr': z_arr,
        'ground_truth': gt_2d,
        'geometry_matrix': L,
        'measurements_clean': y_clean,
        'measurements_noisy': y_noisy,
        'n_los': n_los,
        'nr': nr,
        'nz': nz,
        'n_detectors': n_detectors,
        'n_los_per_det': n_los_per_det
    }
