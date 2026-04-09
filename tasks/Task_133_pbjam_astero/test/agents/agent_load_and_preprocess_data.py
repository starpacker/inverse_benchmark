import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import uniform_filter1d, percentile_filter

def load_and_preprocess_data(
    nu_max=2200.0,
    delta_nu=103.0,
    epsilon=1.45,
    d02=7.5,
    n_orders=8,
    l_max=2,
    linewidth=1.2,
    freq_min=1500.0,
    freq_max=3000.0,
    freq_resolution=0.1,
    bg_params=(5.0, 800.0, 2.0, 0.5),
    base_height=50.0,
    seed=42
):
    """
    Generate synthetic stellar power spectrum with known mode frequencies.
    
    Returns:
        noisy_spectrum: ndarray, the noisy observed power spectrum
        gt_freq_array: ndarray, ground truth mode frequencies
        metadata: dict containing all auxiliary data
    """
    np.random.seed(seed)
    
    freqs = np.arange(freq_min, freq_max, freq_resolution)
    
    # Compute ground truth mode frequencies using asymptotic relation
    n_center = int(round(nu_max / delta_nu - epsilon))
    n_start = n_center - n_orders // 2
    n_end = n_start + n_orders
    
    D0 = d02 / 6.0
    
    gt_modes = {}
    for n in range(n_start, n_end):
        for l in range(l_max + 1):
            freq = delta_nu * (n + l / 2.0 + epsilon) - l * (l + 1) * D0
            if freq_min < freq < freq_max:
                gt_modes[(n, l)] = freq
    
    # Compute mode heights with Gaussian envelope
    visibility = {0: 1.0, 1: 1.5, 2: 0.5}
    env_width = 0.66 * delta_nu * n_orders / 2.0
    
    heights = {}
    for (n, l), freq in gt_modes.items():
        envelope = np.exp(-0.5 * ((freq - nu_max) / env_width) ** 2)
        V_l = visibility.get(l, 0.3)
        heights[(n, l)] = base_height * envelope * V_l ** 2
    
    # Generate clean power spectrum using forward model
    A, nu_c, alpha, W = bg_params
    background = A / (1.0 + (freqs / nu_c) ** alpha) + W
    
    clean_spectrum = np.zeros_like(freqs)
    for (n, l), freq in gt_modes.items():
        H = heights.get((n, l), 1.0)
        gamma = linewidth
        clean_spectrum += H / (1.0 + ((freqs - freq) / gamma) ** 2)
    clean_spectrum += background
    
    # Add chi-squared noise (2 DOF for power spectrum)
    noise_factor = np.random.exponential(scale=1.0, size=len(freqs))
    noisy_spectrum = clean_spectrum * noise_factor
    
    # Smooth slightly for peak detection
    smoothed_spectrum = uniform_filter1d(noisy_spectrum, size=5)
    
    # Prepare GT as sorted frequency array
    gt_freq_list = sorted(gt_modes.values())
    gt_freq_array = np.array(gt_freq_list)
    
    metadata = {
        'freqs': freqs,
        'gt_modes': gt_modes,
        'heights': heights,
        'bg_params': bg_params,
        'clean_spectrum': clean_spectrum,
        'background': background,
        'smoothed_spectrum': smoothed_spectrum,
        'linewidth': linewidth,
        'delta_nu': delta_nu,
        'nu_max': nu_max,
        'freq_resolution': freq_resolution,
        'freq_min': freq_min,
        'freq_max': freq_max,
    }
    
    print(f"[DATA] Generated {len(gt_modes)} p-modes across l=0,1,2")
    print(f"[DATA] Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} μHz")
    print(f"[DATA] Δν = {delta_nu} μHz, ν_max = {nu_max} μHz")
    print(f"[DATA] GT frequencies: {gt_freq_array}")
    
    return noisy_spectrum, gt_freq_array, metadata
