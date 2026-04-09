import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(reflectivity, x_traces, z_depth, dt, nt, v_em, wavelet):
    """
    Exploding reflector forward model for GPR B-scan synthesis.
    
    d(x,t) = ∫∫ r(x',z) · w(t - 2√((x-x')²+z²)/v) dx' dz
    
    Args:
        reflectivity: subsurface reflectivity model (nx, nz)
        x_traces: trace positions array
        z_depth: depth axis array
        dt: time sampling interval
        nt: number of time samples
        v_em: EM wave velocity [m/ns]
        wavelet: source wavelet array
    
    Returns:
        bscan: predicted B-scan data (nx, nt)
    """
    nx = len(x_traces)
    nz = len(z_depth)
    bscan = np.zeros((nx, nt))
    dx_scene = x_traces[1] - x_traces[0] if len(x_traces) > 1 else 0.05
    n_wav = len(wavelet)
    
    # Get non-zero reflector positions
    nz_idx = np.argwhere(reflectivity > 1e-10)
    
    if len(nz_idx) == 0:
        return bscan
    
    r_vals = reflectivity[nz_idx[:, 0], nz_idx[:, 1]]
    x_refl = nz_idx[:, 0].astype(float) * dx_scene
    z_refl = z_depth[nz_idx[:, 1]]
    
    for ix_t in range(nx):
        x_recv = x_traces[ix_t]
        # Vectorized over all reflectors
        dist = np.sqrt((x_recv - x_refl)**2 + z_refl**2)
        twt = 2 * dist / (v_em * 1e9)
        it_arr = (twt / dt).astype(int)
        
        valid = (it_arr >= 0) & (it_arr < nt)
        for k in np.where(valid)[0]:
            it = it_arr[k]
            it_start = max(0, it - n_wav // 2)
            it_end = min(nt, it + n_wav // 2)
            wav_start = it_start - (it - n_wav // 2)
            wav_end = wav_start + (it_end - it_start)
            bscan[ix_t, it_start:it_end] += r_vals[k] * wavelet[wav_start:wav_end]
    
    return bscan
