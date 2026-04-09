import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

from scipy.stats import gamma as gamma_dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

EPS_HBO_760 = 1486.5865

EPS_HBR_760 = 3843.707

EPS_HBO_850 = 2526.391

EPS_HBR_850 = 1798.643

DPF_760 = 6.0

DPF_850 = 5.5

D = 3.0

def canonical_hrf(t_hrf, peak=6.0, undershoot=16.0, ratio=6.0):
    """Generate a canonical hemodynamic response function (double-gamma)."""
    h = (gamma_dist.pdf(t_hrf, peak / 1.0, scale=1.0) -
         gamma_dist.pdf(t_hrf, undershoot / 1.0, scale=1.0) / ratio)
    h = h / np.max(np.abs(h))
    return h

def load_and_preprocess_data(fs, duration, snr_db=25, seed=42):
    """
    Synthesize ground truth hemodynamic signals (HbO, HbR) with block design,
    apply forward MBLL to generate optical density, and add noise.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    duration : float
        Total duration in seconds
    snr_db : float
        Signal-to-noise ratio in dB
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 't': time vector
        - 'hbo_gt': ground truth HbO concentration
        - 'hbr_gt': ground truth HbR concentration
        - 'od_760_noisy': noisy optical density at 760nm
        - 'od_850_noisy': noisy optical density at 850nm
        - 'block_starts': list of stimulus block start times
        - 'block_duration': duration of each stimulus block
        - 'fs': sampling frequency
    """
    rng = np.random.default_rng(seed)
    
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Create stimulus blocks: 3 blocks, each ~20s on, ~60s off
    stimulus = np.zeros(n_samples)
    block_starts = [30, 110, 200]
    block_duration = 20
    for start in block_starts:
        i0 = int(start * fs)
        i1 = int((start + block_duration) * fs)
        stimulus[i0:min(i1, n_samples)] = 1.0
    
    # HRF kernel (30 seconds long)
    t_hrf = np.arange(0, 30, 1.0 / fs)
    hrf = canonical_hrf(t_hrf)
    
    # Convolve stimulus with HRF
    bold = np.convolve(stimulus, hrf, mode='full')[:n_samples]
    bold = bold / np.max(np.abs(bold))
    
    # HbO: positive response, peak ~5 µM
    hbo_gt = bold * 5e-6  # in Molar
    
    # HbR: negative, smaller (~-1.5 µM)
    hbr_gt = -bold * 1.5e-6
    
    # Apply forward MBLL to get clean optical density
    od_760_clean = (EPS_HBO_760 * hbo_gt * DPF_760 * D +
                    EPS_HBR_760 * hbr_gt * DPF_760 * D)
    od_850_clean = (EPS_HBO_850 * hbo_gt * DPF_850 * D +
                    EPS_HBR_850 * hbr_gt * DPF_850 * D)
    
    # Add noise
    def add_noise(signal, snr):
        sig_power = np.mean(signal ** 2)
        noise_power = sig_power / (10 ** (snr / 10))
        noise = rng.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    od_760_noisy = add_noise(od_760_clean, snr_db)
    od_850_noisy = add_noise(od_850_clean, snr_db)
    
    data_dict = {
        't': t,
        'hbo_gt': hbo_gt,
        'hbr_gt': hbr_gt,
        'od_760_noisy': od_760_noisy,
        'od_850_noisy': od_850_noisy,
        'block_starts': block_starts,
        'block_duration': block_duration,
        'fs': fs
    }
    
    return data_dict
