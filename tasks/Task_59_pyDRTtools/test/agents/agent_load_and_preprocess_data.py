import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(n_freq, freq_min, freq_max, n_tau, tau_min, tau_max,
                              gt_peaks, r_inf, r_pol, noise_level, seed):
    """
    Generate synthetic EIS data from a known DRT distribution.
    
    Parameters
    ----------
    n_freq : int
        Number of frequency points.
    freq_min : float
        Minimum frequency [Hz].
    freq_max : float
        Maximum frequency [Hz].
    n_tau : int
        Number of relaxation time points.
    tau_min : float
        Minimum relaxation time [s].
    tau_max : float
        Maximum relaxation time [s].
    gt_peaks : list of dict
        Ground truth DRT peaks, each with 'tau_center', 'gamma_max', 'sigma'.
    r_inf : float
        High-frequency resistance [Ohm].
    r_pol : float
        Polarisation resistance [Ohm].
    noise_level : float
        Relative noise level on impedance.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'freq': frequency array [Hz]
        - 'tau': relaxation time array [s]
        - 'gamma_gt': ground truth DRT
        - 'Z_clean': clean impedance
        - 'Z_noisy': noisy impedance
        - 'r_inf': high-frequency resistance
        - 'r_pol': polarisation resistance
    """
    print("[DATA] Generating synthetic EIS from DRT ...")
    
    # Create frequency and tau grids
    freq = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    
    # Generate ground truth DRT as sum of Gaussians in log(tau)
    ln_tau = np.log(tau)
    gamma_gt = np.zeros_like(tau)
    for p in gt_peaks:
        ln_center = np.log(p["tau_center"])
        gamma_gt += p["gamma_max"] * np.exp(
            -(ln_tau - ln_center) ** 2 / (2 * p["sigma"] ** 2)
        )
    
    # Normalize so integral ~ 1
    d_ln_tau = np.diff(ln_tau)
    integral = np.sum(0.5 * (gamma_gt[:-1] + gamma_gt[1:]) * d_ln_tau)
    if integral > 0:
        gamma_gt = gamma_gt / integral
    
    # Compute clean impedance using forward operator
    Z_clean = forward_operator(gamma_gt, tau, freq, r_inf, r_pol)
    
    # Add noise
    rng = np.random.default_rng(seed)
    Z_mag = np.abs(Z_clean)
    noise = noise_level * Z_mag * (
        rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)
    )
    Z_noisy = Z_clean + noise
    
    print(f"[DATA] {n_freq} frequencies: [{freq_min:.0e}, {freq_max:.0e}] Hz")
    print(f"[DATA] |Z| range: [{Z_mag.min():.2f}, {Z_mag.max():.2f}] Ω")
    print(f"[DATA] DRT peaks at τ = {[p['tau_center'] for p in gt_peaks]}")
    
    data_dict = {
        'freq': freq,
        'tau': tau,
        'gamma_gt': gamma_gt,
        'Z_clean': Z_clean,
        'Z_noisy': Z_noisy,
        'r_inf': r_inf,
        'r_pol': r_pol,
    }
    
    return data_dict

def forward_operator(gamma, tau, freq, r_inf, r_pol):
    """
    Compute EIS impedance from DRT via Fredholm integral.
    
    Z(ω) = R_∞ + R_pol ∫ γ(τ)/(1 + iωτ) d(ln τ)
    
    Parameters
    ----------
    gamma : np.ndarray
        DRT values γ(τ).
    tau : np.ndarray
        Relaxation times [s].
    freq : np.ndarray
        Frequencies [Hz].
    r_inf : float
        High-frequency resistance [Ω].
    r_pol : float
        Polarisation resistance [Ω].
    
    Returns
    -------
    Z : np.ndarray
        Complex impedance [Ω].
    """
    omega = 2 * np.pi * freq
    ln_tau = np.log(tau)
    
    # Compute d(ln τ) for integration
    d_ln_tau = np.zeros_like(ln_tau)
    d_ln_tau[1:-1] = (ln_tau[2:] - ln_tau[:-2]) / 2
    d_ln_tau[0] = ln_tau[1] - ln_tau[0]
    d_ln_tau[-1] = ln_tau[-1] - ln_tau[-2]
    
    Z = np.full(len(freq), r_inf, dtype=complex)
    for i, w in enumerate(omega):
        integrand = gamma / (1 + 1j * w * tau)
        Z[i] += r_pol * np.sum(integrand * d_ln_tau)
    
    return Z
