import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(nx_traces, nz, nt, dx, dt, z_max, v_em, freq_center, noise_snr_db, seed):
    """
    Generate synthetic subsurface model and GPR B-scan data.
    
    Returns:
        dict containing all necessary data for inversion:
            - reflectivity: ground truth subsurface model
            - bscan_noisy: noisy B-scan measurements
            - bscan_clean: clean B-scan (for reference)
            - x_traces: trace positions
            - z_depth: depth axis
            - t_axis: time axis
            - dt, v_em: acquisition parameters
    """
    rng = np.random.default_rng(seed)
    
    # Create depth and position axes
    z = np.linspace(0, z_max, nz)
    x = np.arange(nx_traces) * dx
    
    # Generate subsurface reflectivity model
    reflectivity = np.zeros((nx_traces, nz))
    
    # Layer 1: horizontal at z = 0.5 m
    iz1 = np.argmin(np.abs(z - 0.5))
    reflectivity[:, iz1] = 0.3
    
    # Layer 2: dipping interface
    for ix in range(nx_traces):
        z_dip = 1.0 + 0.3 * (ix / nx_traces)
        iz_dip = np.argmin(np.abs(z - z_dip))
        if iz_dip < nz:
            reflectivity[ix, iz_dip] = 0.5
    
    # Layer 3: horizontal at z = 1.8 m
    iz3 = np.argmin(np.abs(z - 1.8))
    reflectivity[:, iz3] = 0.4
    
    # Point diffractors
    diffractors = [
        (nx_traces // 4, 0.8, 0.7),
        (nx_traces // 2, 1.3, 0.6),
        (3 * nx_traces // 4, 0.6, 0.8),
    ]
    for ix, z_d, amp in diffractors:
        iz_d = np.argmin(np.abs(z - z_d))
        if 0 <= ix < nx_traces and 0 <= iz_d < nz:
            reflectivity[ix, iz_d] = amp
    
    # Small void (rectangular)
    ix_void_start = int(0.6 * nx_traces)
    ix_void_end = int(0.65 * nx_traces)
    iz_void = np.argmin(np.abs(z - 1.1))
    iz_void_end = np.argmin(np.abs(z - 1.3))
    reflectivity[ix_void_start:ix_void_end, iz_void] = 0.6
    reflectivity[ix_void_start:ix_void_end, iz_void_end] = 0.6
    reflectivity[ix_void_start, iz_void:iz_void_end] = 0.5
    reflectivity[ix_void_end, iz_void:iz_void_end] = 0.5
    
    # Generate Ricker wavelet
    n_wav = 64
    t_wav = np.arange(n_wav) * dt
    t_centre = t_wav[n_wav // 2]
    tau = t_wav - t_centre
    sigma = 1.0 / (np.pi * freq_center * np.sqrt(2))
    wavelet = (1 - (tau / sigma)**2) * np.exp(-0.5 * (tau / sigma)**2)
    wavelet /= np.max(np.abs(wavelet))
    
    # Generate B-scan using exploding reflector model
    bscan_clean = np.zeros((nx_traces, nt))
    dx_scene = dx
    
    # Get non-zero reflector positions
    nz_idx = np.argwhere(reflectivity > 1e-10)
    
    if len(nz_idx) > 0:
        r_vals = reflectivity[nz_idx[:, 0], nz_idx[:, 1]]
        x_refl = nz_idx[:, 0].astype(float) * dx_scene
        z_refl = z[nz_idx[:, 1]]
        
        for ix_t in range(nx_traces):
            x_recv = x[ix_t]
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
                bscan_clean[ix_t, it_start:it_end] += r_vals[k] * wavelet[wav_start:wav_end]
    
    # Add noise
    sig_power = np.mean(bscan_clean**2)
    noise_power = sig_power / (10**(noise_snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(bscan_clean.shape)
    bscan_noisy = bscan_clean + noise
    
    t_axis = np.arange(nt) * dt
    
    data = {
        'reflectivity': reflectivity,
        'bscan_noisy': bscan_noisy,
        'bscan_clean': bscan_clean,
        'x_traces': x,
        'z_depth': z,
        't_axis': t_axis,
        'dt': dt,
        'v_em': v_em,
        'wavelet': wavelet
    }
    
    return data
