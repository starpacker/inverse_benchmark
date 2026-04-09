import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

from scipy.ndimage import uniform_filter, uniform_filter1d

def load_and_preprocess_data(nt, n_traces, dt, freq_dominant, snr_db, seed=42):
    """
    Load and preprocess seismic data.
    
    Creates:
    - Ricker wavelet
    - Synthetic 2D acoustic impedance model
    - Forward-modeled seismic data with added noise
    
    Returns:
        dict containing:
            - wavelet: Ricker wavelet array
            - wav_t: wavelet time axis
            - impedance_true: true impedance model (nt, n_traces)
            - reflectivity_true: true reflectivity (nt, n_traces)
            - seismic_obs: observed (noisy) seismic data (nt, n_traces)
            - seismic_clean: clean seismic data (nt, n_traces)
            - params: dictionary of parameters
    """
    np.random.seed(seed)
    
    # Create Ricker wavelet
    length = 0.128
    t_wav = np.arange(-length/2, length/2 + dt, dt)
    t2 = t_wav**2
    sigma2 = 1.0 / (np.pi * freq_dominant)**2
    wavelet = (1.0 - 2*np.pi**2 * freq_dominant**2 * t2) * np.exp(-np.pi**2 * freq_dominant**2 * t2)
    
    print(f"[WAV] Wavelet length: {len(wavelet)} samples")
    
    # Generate synthetic earth model
    t = np.arange(nt) * dt
    
    # Base impedance model (layered structure)
    impedance_1d = np.ones(nt) * 2500.0  # Base impedance (kg/m²/s × 10³)
    
    # Add layers at different depths
    layers = [
        (80,  120,  3200.0),   # Layer 1: sandstone
        (150, 180,  2800.0),   # Layer 2: shale
        (200, 260,  3800.0),   # Layer 3: limestone
        (280, 320,  3000.0),   # Layer 4: sandstone
        (340, 380,  4200.0),   # Layer 5: dolomite
        (400, 430,  3500.0),   # Layer 6: tight sand
    ]
    
    for top, bot, imp in layers:
        impedance_1d[top:bot] = imp
    
    # Create 2D model with lateral variations
    impedance_true = np.zeros((nt, n_traces))
    for j in range(n_traces):
        offset = int(5 * np.sin(2 * np.pi * j / n_traces))  # Gentle dipping
        imp_shifted = np.roll(impedance_1d, offset)
        
        # Add lateral impedance variation
        lateral_factor = 1.0 + 0.05 * np.sin(2 * np.pi * j / (n_traces / 2))
        impedance_true[:, j] = imp_shifted * lateral_factor
    
    # Compute reflectivity from impedance contrasts
    # r(t) = (I(t+1) - I(t)) / (I(t+1) + I(t))
    reflectivity_true = np.zeros_like(impedance_true)
    reflectivity_true[1:, :] = (impedance_true[1:, :] - impedance_true[:-1, :]) / \
                               (impedance_true[1:, :] + impedance_true[:-1, :])
    
    print(f"  Time axis: {nt} samples × {dt*1000:.0f} ms = {nt*dt:.2f} s")
    print(f"  Traces: {n_traces}")
    print(f"  Impedance range: [{impedance_true.min():.0f}, {impedance_true.max():.0f}]")
    print(f"  Reflectivity range: [{reflectivity_true.min():.4f}, {reflectivity_true.max():.4f}]")
    print(f"  Non-zero reflectors: {np.sum(np.abs(reflectivity_true) > 0.001)}")
    
    # Forward model: compute seismic data
    seismic_clean = forward_operator(reflectivity_true, wavelet)
    print(f"[FWD] Seismic range: [{seismic_clean.min():.6f}, {seismic_clean.max():.6f}]")
    
    # Add noise
    signal_power = np.mean(seismic_clean**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(*seismic_clean.shape)
    seismic_obs = seismic_clean + noise
    print(f"[NOISE] Added noise (SNR={snr_db} dB)")
    
    # Create smooth background impedance model
    impedance_bg = uniform_filter1d(impedance_true, size=50, axis=0)
    
    params = {
        'nt': nt,
        'n_traces': n_traces,
        'dt': dt,
        'freq_dominant': freq_dominant,
        'snr_db': snr_db,
    }
    
    return {
        'wavelet': wavelet,
        'wav_t': t_wav,
        'impedance_true': impedance_true,
        'reflectivity_true': reflectivity_true,
        'seismic_obs': seismic_obs,
        'seismic_clean': seismic_clean,
        'impedance_bg': impedance_bg,
        'params': params,
    }

def forward_operator(reflectivity, wavelet):
    """
    Forward operator: Convolve reflectivity with source wavelet.
    s(t) = w(t) * r(t) using pylops Convolve1D operator.
    
    Args:
        reflectivity: reflectivity array (nt,) or (nt, n_traces)
        wavelet: source wavelet array
    
    Returns:
        seismic: convolved seismic data, same shape as reflectivity
    """
    from pylops.signalprocessing import Convolve1D
    
    # Handle both 1D and 2D cases
    if reflectivity.ndim == 1:
        nt = reflectivity.shape[0]
        Cop = Convolve1D(nt, h=wavelet, offset=len(wavelet)//2)
        seismic = Cop @ reflectivity
    else:
        nt, n_traces = reflectivity.shape
        seismic = np.zeros_like(reflectivity)
        
        for j in range(n_traces):
            # Create convolution operator
            Cop = Convolve1D(nt, h=wavelet, offset=len(wavelet)//2)
            seismic[:, j] = Cop @ reflectivity[:, j]
    
    return seismic
