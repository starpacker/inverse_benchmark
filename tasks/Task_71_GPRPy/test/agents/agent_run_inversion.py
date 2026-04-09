import numpy as np

import matplotlib

matplotlib.use('Agg')

def run_inversion(bscan, x_traces, z_depth, dt, v_em):
    """
    Kirchhoff depth migration for GPR data inversion.
    
    Recovers subsurface reflectivity from B-scan using time-domain
    summation along hyperbolic travel-time curves:
    
    r̂(x,z) = Σ_x' d(x', t=2R/v) · cos(θ)/R
    
    Args:
        bscan: measured B-scan data (nx, nt)
        x_traces: trace positions array
        z_depth: depth axis array  
        dt: time sampling interval
        v_em: EM wave velocity [m/ns]
    
    Returns:
        migrated: reconstructed reflectivity image (nx, nz)
    """
    nx = len(x_traces)
    nz = len(z_depth)
    nt = bscan.shape[1]
    image = np.zeros((nx, nz))
    
    x_tr_arr = np.asarray(x_traces)
    
    for ix_img in range(nx):
        x_pt = x_traces[ix_img]
        for iz_img in range(nz):
            z_pt = z_depth[iz_img]
            if z_pt < 1e-6:
                continue
            
            # Vectorized over all traces
            R = np.sqrt((x_pt - x_tr_arr)**2 + z_pt**2)
            twt = 2 * R / (v_em * 1e9)
            it_float = twt / dt
            it_int = np.floor(it_float).astype(int)
            frac = it_float - it_int
            
            valid = (it_int >= 0) & (it_int < nt - 1)
            it_safe = np.clip(it_int, 0, nt - 2)
            
            # Linear interpolation
            d_interp = (1 - frac) * bscan[np.arange(nx), it_safe] + \
                       frac * bscan[np.arange(nx), it_safe + 1]
            
            # Obliquity and spreading correction
            cos_theta = z_pt / np.maximum(R, 1e-10)
            weight = cos_theta / np.maximum(R, 1e-6)
            
            image[ix_img, iz_img] = np.sum(valid * d_interp * weight)
    
    # Clip negative artifacts (reflectivity is non-negative)
    migrated = np.clip(image, 0, None)
    
    return migrated
