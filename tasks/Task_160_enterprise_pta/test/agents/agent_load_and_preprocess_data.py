import os

import numpy as np

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def hellings_downs(theta):
    """Hellings-Downs overlap reduction function."""
    x = (1.0 - np.cos(theta)) / 2.0
    if x < 1e-10:
        return 1.0
    hd = 1.5 * x * np.log(x) - 0.25 * x + 0.5
    return hd

def make_hd_matrix(positions):
    """Build the N_psr x N_psr Hellings-Downs correlation matrix."""
    n = len(positions)
    hd = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = np.dot(positions[i], positions[j])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            val = hellings_downs(theta)
            hd[i, j] = val
            hd[j, i] = val
    return hd

def powerlaw_psd(freqs, log10_A, gamma):
    """Power-law power spectral density: S(f) = A^2/(12*pi^2) * (f/f_yr)^(-gamma) * f_yr^(-3)."""
    A = 10.0 ** log10_A
    f_yr = 1.0 / (365.25 * 86400.0)
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs / f_yr) ** (-gamma) * f_yr ** (-3)

def fourier_design_matrix(toas, n_freq, T):
    """Create the Fourier design matrix F (N_toa x 2*n_freq)."""
    N = len(toas)
    F = np.zeros((N, 2 * n_freq))
    freqs = np.arange(1, n_freq + 1) / T
    for i, f in enumerate(freqs):
        F[:, 2 * i] = np.sin(2.0 * np.pi * f * toas)
        F[:, 2 * i + 1] = np.cos(2.0 * np.pi * f * toas)
    return F, freqs

def load_and_preprocess_data(n_pulsars, n_toa, t_span_yr, n_freq,
                              true_log10_A_gw, true_log10_A_red, true_gamma_red,
                              true_efac, true_sigma_wn):
    """
    Simulate and preprocess PTA data.
    
    Returns:
        data_dict: Dictionary containing all necessary data for inference
    """
    t_span = t_span_yr * 365.25 * 86400.0  # seconds
    
    # Random sky positions (unit vectors)
    positions = []
    for _ in range(n_pulsars):
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        positions.append(np.array([sin_theta * np.cos(phi),
                                   sin_theta * np.sin(phi),
                                   cos_theta]))

    hd_matrix = make_hd_matrix(positions)
    freqs = np.arange(1, n_freq + 1) / t_span

    # Red noise PSD per frequency
    psd_red = powerlaw_psd(freqs, true_log10_A_red, true_gamma_red)
    # GW PSD per frequency (GWB spectral index = 13/3 for circular binaries)
    psd_gw = powerlaw_psd(freqs, true_log10_A_gw, 13.0 / 3.0)

    toas_all = []
    residuals_all = []
    F_all = []

    for p in range(n_pulsars):
        # Uniform TOAs
        toas = np.linspace(0, t_span, n_toa)
        toas_all.append(toas)

        F, _ = fourier_design_matrix(toas, n_freq, t_span)
        F_all.append(F)

    # Generate correlated GW Fourier coefficients across pulsars
    gw_coeffs = np.zeros((n_pulsars, 2 * n_freq))
    for k in range(n_freq):
        amplitude = np.sqrt(psd_gw[k] * t_span)
        # Cholesky of HD matrix for correlation
        L = np.linalg.cholesky(hd_matrix + 1e-10 * np.eye(n_pulsars))
        for c in range(2):  # sin and cos
            uncorr = np.random.randn(n_pulsars)
            corr = L @ uncorr * amplitude
            gw_coeffs[:, 2 * k + c] = corr

    for p in range(n_pulsars):
        F = F_all[p]
        toas = toas_all[p]

        # Red noise (independent per pulsar)
        red_coeffs = np.zeros(2 * n_freq)
        for k in range(n_freq):
            amplitude = np.sqrt(psd_red[k] * t_span)
            red_coeffs[2 * k] = np.random.randn() * amplitude
            red_coeffs[2 * k + 1] = np.random.randn() * amplitude

        # Total signal
        signal = F @ (red_coeffs + gw_coeffs[p])

        # White noise
        wn = np.random.randn(n_toa) * true_sigma_wn * true_efac

        residuals = signal + wn
        residuals_all.append(residuals)

    # Package all data into a dictionary
    data_dict = {
        'toas_all': toas_all,
        'residuals_all': residuals_all,
        'F_all': F_all,
        'freqs': freqs,
        'hd_matrix': hd_matrix,
        'positions': positions,
        't_span': t_span,
        'n_pulsars': n_pulsars,
        'n_toa': n_toa,
        'n_freq': n_freq,
        'sigma_wn': true_sigma_wn,
        'efac': true_efac,
        'true_params': np.array([true_log10_A_gw, true_log10_A_red, true_gamma_red]),
        'psd_red_true': psd_red,
        'psd_gw_true': psd_gw,
    }
    
    return data_dict
