import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(
    n_freq,
    n_angle,
    freq_min,
    freq_max,
    freq_ref,
    t21_rms,
    t_fg_amp,
    beta_mean,
    beta_std,
    noise_rms,
    seed
):
    """
    Generate and preprocess all data for the 21cm tomography problem.
    
    Returns:
        dict containing:
            - frequencies: frequency grid array
            - T21_gt: ground truth 21cm signal
            - T_fg: foreground in Kelvin
            - T_fg_mK: foreground in milliKelvin
            - observation: simulated observation (signal + foreground + noise)
            - noise: noise realization
            - params: dictionary of simulation parameters
    """
    np.random.seed(seed)
    
    # Create frequency grid
    frequencies = np.linspace(freq_min, freq_max, n_freq)
    
    # Generate 21cm signal with spectral and angular correlations
    raw = np.random.randn(n_freq, n_angle)
    
    # Spectral correlation kernel
    fk = max(5, n_freq // 16)
    freq_kernel = np.exp(-0.5 * np.linspace(-2.5, 2.5, fk)**2)
    freq_kernel /= freq_kernel.sum()
    
    # Angular correlation kernel
    ak = max(5, n_angle // 8)
    ang_kernel = np.exp(-0.5 * np.linspace(-2.5, 2.5, ak)**2)
    ang_kernel /= ang_kernel.sum()
    
    T21 = np.zeros_like(raw)
    for j in range(n_angle):
        T21[:, j] = np.convolve(raw[:, j], freq_kernel, mode='same')
    for i in range(n_freq):
        T21[i, :] = np.convolve(T21[i, :], ang_kernel, mode='same')
    
    T21 -= T21.mean()
    T21 = T21 / np.std(T21) * t21_rms
    T21_gt = T21
    
    # Generate astrophysical foreground (synchrotron power-law)
    # T_fg(ν,θ) = A(θ) * (ν/ν_ref)^{-β(θ)}
    angles_norm = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    beta_spatial = beta_mean + beta_std * np.sin(angles_norm * 2)
    amp_spatial = t_fg_amp * (1.0 + 0.2 * np.cos(angles_norm * 3))
    
    freq_ratio = frequencies[:, np.newaxis] / freq_ref
    T_fg = amp_spatial[np.newaxis, :] * freq_ratio ** (-beta_spatial[np.newaxis, :])
    T_fg_mK = T_fg * 1000.0
    
    # Generate noise
    noise = noise_rms * np.random.randn(n_freq, n_angle)
    
    # Create observation: y = T_21 + T_fg*1000 + noise (all in mK)
    observation = T21_gt + T_fg_mK + noise
    
    params = {
        'n_freq': n_freq,
        'n_angle': n_angle,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'freq_ref': freq_ref,
        't21_rms': t21_rms,
        't_fg_amp': t_fg_amp,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'noise_rms': noise_rms,
        'seed': seed
    }
    
    return {
        'frequencies': frequencies,
        'T21_gt': T21_gt,
        'T_fg': T_fg,
        'T_fg_mK': T_fg_mK,
        'observation': observation,
        'noise': noise,
        'params': params
    }
