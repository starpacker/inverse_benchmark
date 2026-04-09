import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(n=128, n_ant=16, n_hour=12, max_baseline=None, noise_level=0.02, seed=42):
    """
    Generate synthetic solar radio source model and simulate interferometric observations.
    
    Returns
    -------
    data : dict containing:
        - model: ground truth solar brightness model
        - u, v: uv coordinates
        - visibilities: measured complex visibilities
        - valid: boolean mask for valid visibilities
        - config: configuration parameters
    """
    np.random.seed(seed)
    
    if max_baseline is None:
        max_baseline = n * 0.4
    
    # Generate solar model
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    model = np.zeros((n, n))
    
    # Solar disk (quiet Sun) - large Gaussian
    r2 = X**2 + Y**2
    solar_radius = 0.4
    model += 1.0 * np.exp(-r2 / (2 * solar_radius**2))
    
    # Apply sharp solar limb
    limb_mask = r2 < (0.55)**2
    model *= limb_mask
    
    # Active region 1 - bright compact source (flare)
    cx1, cy1 = 0.15, 0.2
    sigma1 = 0.04
    model += 3.0 * np.exp(-((X - cx1)**2 + (Y - cy1)**2) / (2 * sigma1**2))
    
    # Active region 2 - moderate source
    cx2, cy2 = -0.2, -0.1
    sigma2 = 0.06
    model += 2.0 * np.exp(-((X - cx2)**2 + (Y - cy2)**2) / (2 * sigma2**2))
    
    # Active region 3 - small bright source
    cx3, cy3 = 0.25, -0.15
    sigma3 = 0.03
    model += 4.0 * np.exp(-((X - cx3)**2 + (Y - cy3)**2) / (2 * sigma3**2))
    
    # Ensure non-negative
    model = np.maximum(model, 0)
    
    # Generate UV coverage (Y-shaped array like VLA)
    ant_x = np.zeros(n_ant)
    ant_y = np.zeros(n_ant)
    
    n_per_arm = n_ant // 3
    for arm in range(3):
        angle = arm * 2 * np.pi / 3
        for i in range(n_per_arm):
            idx = arm * n_per_arm + i
            r = max_baseline * (i + 1) / n_per_arm * 0.5
            ant_x[idx] = r * np.cos(angle)
            ant_y[idx] = r * np.sin(angle)
    
    # Remaining antennas at center
    for i in range(3 * n_per_arm, n_ant):
        ant_x[i] = np.random.uniform(-2, 2)
        ant_y[i] = np.random.uniform(-2, 2)
    
    # Generate baselines for all antenna pairs with Earth rotation synthesis
    u_list = []
    v_list = []
    
    hour_angles = np.linspace(-np.pi/3, np.pi/3, n_hour)
    declination = np.radians(20)
    
    for ha in hour_angles:
        cos_ha = np.cos(ha)
        sin_ha = np.sin(ha)
        
        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                bx = ant_x[j] - ant_x[i]
                by = ant_y[j] - ant_y[i]
                
                # Project baseline onto (u,v) plane
                u = bx * sin_ha + by * cos_ha
                v = -bx * cos_ha * np.sin(declination) + by * sin_ha * np.sin(declination)
                
                u_list.append(u)
                v_list.append(v)
                # Add conjugate (symmetry)
                u_list.append(-u)
                v_list.append(-v)
    
    u = np.array(u_list)
    v = np.array(v_list)
    
    # Simulate visibility measurements: V(u,v) = FT{I}(u,v) + noise
    model_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(model)))
    
    # Sample at (u,v) points by nearest-neighbor interpolation
    u_pix = np.round(u + n // 2).astype(int)
    v_pix = np.round(v + n // 2).astype(int)
    
    # Clip to valid range
    valid = (u_pix >= 0) & (u_pix < n) & (v_pix >= 0) & (v_pix < n)
    u_pix_valid = u_pix[valid]
    v_pix_valid = v_pix[valid]
    
    n_vis = len(u)
    visibilities_clean = np.zeros(n_vis, dtype=complex)
    visibilities_clean[valid] = model_fft[v_pix_valid, u_pix_valid]
    
    # Add complex noise
    signal_rms = np.sqrt(np.mean(np.abs(visibilities_clean[valid])**2))
    noise = noise_level * signal_rms * (
        np.random.randn(n_vis) + 1j * np.random.randn(n_vis)
    ) / np.sqrt(2)
    
    visibilities = visibilities_clean + noise
    
    config = {
        'n': n,
        'n_ant': n_ant,
        'n_hour': n_hour,
        'max_baseline': max_baseline,
        'noise_level': noise_level
    }
    
    data = {
        'model': model,
        'u': u,
        'v': v,
        'visibilities': visibilities,
        'valid': valid,
        'config': config
    }
    
    return data
