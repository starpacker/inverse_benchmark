import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_104_bisip_sip"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def load_and_preprocess_data(n_freq, freq_min, freq_max, gt_params_list, noise_level, seed):
    """
    Generate frequency array and synthetic noisy complex resistivity data
    from Cole-Cole model for multiple spectra.
    
    Parameters:
        n_freq: int, number of frequency points
        freq_min: float, minimum frequency (Hz)
        freq_max: float, maximum frequency (Hz)
        gt_params_list: list of dicts, ground truth Cole-Cole parameters
        noise_level: float, relative noise level
        seed: int, random seed
    
    Returns:
        freq: ndarray, frequency array
        rho_obs_list: list of ndarray, observed (noisy) complex resistivity
        rho_true_list: list of ndarray, true complex resistivity
        gt_params_list: list of dicts, ground truth parameters
    """
    np.random.seed(seed)
    
    # Generate logarithmically spaced frequency array
    freq = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    
    rho_obs_list = []
    rho_true_list = []
    
    for params in gt_params_list:
        rho0 = params["rho0"]
        m = params["m"]
        tau = params["tau"]
        c = params["c"]
        
        # Compute true Cole-Cole response
        omega = 2.0 * np.pi * freq
        z = (1j * omega * tau) ** c
        rho_true = rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + z)))
        
        # Add relative noise to real and imaginary parts
        noise_re = noise_level * np.abs(rho_true) * np.random.randn(len(freq))
        noise_im = noise_level * np.abs(rho_true) * np.random.randn(len(freq))
        rho_obs = rho_true + noise_re + 1j * noise_im
        
        rho_obs_list.append(rho_obs)
        rho_true_list.append(rho_true)
    
    return freq, rho_obs_list, rho_true_list, gt_params_list
