import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import shift as ndi_shift, uniform_filter, gaussian_filter

def run_inversion(raw_noisy, params):
    """
    Inverse solver: Extract sub-aperture views from raw MLA image and estimate depth.
    
    Args:
        raw_noisy: noisy raw MLA sensor image (2D array)
        params: dictionary containing n_angular, baseline, patch_half, etc.
    
    Returns:
        dict containing:
            - lf_recon: reconstructed light field (4D array)
            - recon_center: center sub-aperture view (2D array)
            - est_depth: estimated depth map (2D array)
    """
    n_ang = params['n_angular']
    baseline = params['baseline']
    patch_half = params['patch_half']

    # Extract sub-apertures from raw MLA image
    s_size = raw_noisy.shape[0] // n_ang
    lf_recon = np.zeros((n_ang, n_ang, s_size, s_size), dtype=np.float64)
    for u in range(n_ang):
        for v in range(n_ang):
            lf_recon[u, v] = raw_noisy[u::n_ang, v::n_ang]

    # Get center sub-aperture
    u_c = (n_ang - 1) // 2
    recon_center = lf_recon[u_c, u_c].copy()

    # Estimate depth via block matching
    size = lf_recon.shape[2]
    u_c_float = (n_ang - 1) / 2.0
    v_c_float = u_c_float
    center = lf_recon[int(u_c_float), int(v_c_float)]

    # Candidate depth values — densely sample the range we know is relevant
    z_candidates = np.linspace(1.5, 6.5, 60)
    n_z = len(z_candidates)

    win = 2 * patch_half + 1
    c_mean = uniform_filter(center, size=win, mode='nearest')
    c_sq_mean = uniform_filter(center ** 2, size=win, mode='nearest')
    c_std = np.sqrt(np.maximum(c_sq_mean - c_mean ** 2, 1e-12))

    cost_vol = np.zeros((n_z, size, size), dtype=np.float64)

    for u in range(n_ang):
        for v in range(n_ang):
            du = u - u_c_float
            dv = v - v_c_float
            if du == 0 and dv == 0:
                continue
            ref = lf_recon[u, v]
            for zi, z_val in enumerate(z_candidates):
                # The view (u,v) was created by shifting the scene by
                # (baseline*du/Z, baseline*dv/Z).
                # To undo, shift the view by the NEGATIVE of that.
                sx = -baseline * du / z_val   # column shift
                sy = -baseline * dv / z_val   # row shift
                warped = ndi_shift(ref, [sy, sx], order=1, mode='nearest')
                w_mean = uniform_filter(warped, size=win, mode='nearest')
                w_sq_mean = uniform_filter(warped ** 2, size=win, mode='nearest')
                w_std = np.sqrt(np.maximum(w_sq_mean - w_mean ** 2, 1e-12))
                cross = uniform_filter(center * warped, size=win, mode='nearest')
                ncc = (cross - c_mean * w_mean) / (c_std * w_std + 1e-12)
                cost_vol[zi] += ncc

    # Winner-take-all
    best_idx = np.argmax(cost_vol, axis=0)
    est_depth = z_candidates[best_idx]

    # Sub-pixel refinement via parabola
    for s in range(size):
        for t in range(size):
            idx = best_idx[s, t]
            if 0 < idx < n_z - 1:
                c0 = cost_vol[idx - 1, s, t]
                c1 = cost_vol[idx, s, t]
                c2 = cost_vol[idx + 1, s, t]
                denom = 2.0 * (c0 - 2 * c1 + c2)
                if abs(denom) > 1e-12:
                    offset = (c0 - c2) / denom
                    refined_idx = idx + np.clip(offset, -0.5, 0.5)
                    est_depth[s, t] = np.interp(refined_idx,
                                                np.arange(n_z), z_candidates)

    # Mild smoothing to denoise
    est_depth = gaussian_filter(est_depth, sigma=1.2)

    return {
        'lf_recon': lf_recon,
        'recon_center': recon_center,
        'est_depth': est_depth,
    }
