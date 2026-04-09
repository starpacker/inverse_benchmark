import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_101_pyilc_cmb"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def gaussian_random_field(N, power_law_index=-2.0, rms=1.0, seed=None):
    """Generate a 2D Gaussian random field with power-law power spectrum."""
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0  # avoid division by zero
    power = K ** power_law_index
    power[0, 0] = 0.0  # zero mean
    phases = np.random.uniform(0, 2 * np.pi, (N, N))
    amplitudes = np.sqrt(power) * np.exp(1j * phases)
    field = np.fft.ifft2(amplitudes).real
    field = field / field.std() * rms
    return field

def load_and_preprocess_data(n_pix, freqs_ghz, cmb_rms, noise_rms, seed):
    """
    Simulate multi-frequency sky maps with CMB, foregrounds, and noise.
    
    Parameters
    ----------
    n_pix : int
        Map side length in pixels.
    freqs_ghz : np.ndarray
        Observation frequencies in GHz.
    cmb_rms : float
        RMS of CMB signal in μK.
    noise_rms : np.ndarray
        Noise RMS per frequency in μK.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    cmb_gt : np.ndarray, shape (n_pix, n_pix)
        Ground truth CMB map.
    data : np.ndarray, shape (n_freq, n_pix, n_pix)
        Multi-frequency observed maps.
    freqs_ghz : np.ndarray
        Frequencies (passed through for convenience).
    """
    np.random.seed(seed)
    
    # Create CMB as Gaussian random field with l^(-2) power spectrum
    cmb_gt = gaussian_random_field(n_pix, power_law_index=-2.0, rms=cmb_rms)
    
    n_freq = len(freqs_ghz)
    data = np.zeros((n_freq, n_pix, n_pix))
    
    for i, nu in enumerate(freqs_ghz):
        # Synchrotron template: power law ∝ ν^{-3.0} from ref_freq=0.408 GHz
        sync_template = gaussian_random_field(n_pix, power_law_index=-2.5, rms=200.0)
        fg_sync = sync_template * (nu / 0.408) ** (-3.0)
        
        # Thermal dust template: modified blackbody ∝ ν^{1.5} from ref_freq=545 GHz
        dust_template = gaussian_random_field(n_pix, power_law_index=-2.5, rms=150.0)
        fg_dust = dust_template * (nu / 545.0) ** 1.5
        
        # Noise
        noise = np.random.normal(0, noise_rms[i], (n_pix, n_pix))
        
        # CMB has flat SED in thermodynamic temperature units: a_ν = 1
        data[i] = cmb_gt + fg_sync + fg_dust + noise
    
    return cmb_gt, data, freqs_ghz
