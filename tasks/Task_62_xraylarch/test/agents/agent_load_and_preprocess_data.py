import numpy as np

import matplotlib

matplotlib.use('Agg')

def feff_amplitude(k, Z):
    """Simplified backscattering amplitude |f(k)|."""
    if Z == 8:  # O
        return 0.5 * np.exp(-0.01 * k**2) * (1 + 0.1 * np.sin(k))
    elif Z == 26:  # Fe
        return 0.8 * np.exp(-0.005 * k**2) * (1 + 0.2 * np.sin(1.5 * k))
    else:
        return 0.6 * np.exp(-0.008 * k**2)

def feff_phase(k, Z):
    """Simplified total phase shift δ(k)."""
    if Z == 8:
        return -0.2 * k + 0.5 + 0.02 * k**2
    elif Z == 26:
        return -0.3 * k + 1.0 + 0.015 * k**2
    else:
        return -0.25 * k + 0.7

def mean_free_path(k):
    """Mean free path λ(k) in Å."""
    return 1.0 / (0.003 * k**2 + 0.01)

def load_and_preprocess_data(gt_shells, k_min, k_max, n_k, noise_level, s02, seed):
    """
    Generate synthetic EXAFS data with ground truth shells.
    
    Parameters
    ----------
    gt_shells : list of dict
        Ground truth shell parameters.
    k_min : float
        Minimum k value [Å^-1].
    k_max : float
        Maximum k value [Å^-1].
    n_k : int
        Number of k-space points.
    noise_level : float
        Standard deviation of Gaussian noise.
    s02 : float
        Amplitude reduction factor.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    k : ndarray
        Photoelectron wavenumber array [Å^-1].
    chi_noisy : ndarray
        Noisy EXAFS oscillation (measured data).
    chi_clean : ndarray
        Clean EXAFS oscillation (ground truth).
    gt_shells : list of dict
        Ground truth shell parameters (passed through).
    """
    print("[DATA] Generating synthetic EXAFS data (Fe-O/Fe-Fe) ...")
    k = np.linspace(k_min, k_max, n_k)
    chi_clean = forward_operator(gt_shells, k, s02)
    
    rng = np.random.default_rng(seed)
    chi_noisy = chi_clean + noise_level * rng.standard_normal(n_k)
    
    print(f"[DATA] k range: [{k_min}, {k_max}] Å⁻¹, {n_k} points")
    print(f"[DATA] χ range: [{chi_clean.min():.4f}, {chi_clean.max():.4f}]")
    print(f"[DATA] Shells: {[s['label'] for s in gt_shells]}")
    
    return k, chi_noisy, chi_clean, gt_shells

def forward_operator(shells, k, s02=0.9):
    """
    Compute EXAFS χ(k) from shell parameters.

    Standard EXAFS equation:
    χ(k) = Σ_j (N_j·S₀²·|f_j(k)|) / (k·R_j²) ·
            sin(2kR_j + δ_j(k)) ·
            exp(-2σ²_j·k²) · exp(-2R_j/λ(k))

    Parameters
    ----------
    shells : list of dict
        Shell parameters with keys: N, R, sigma2, dE0, Z, label.
    k : ndarray
        Photoelectron wavenumber [Å^-1].
    s02 : float
        Amplitude reduction factor.

    Returns
    -------
    chi : ndarray
        EXAFS oscillation function.
    """
    chi = np.zeros_like(k)
    lam = mean_free_path(k)
    
    for sh in shells:
        N = sh["N"]
        R = sh["R"]
        sig2 = sh["sigma2"]
        dE0 = sh.get("dE0", 0)
        Z = sh["Z"]
        
        # Effective k with energy shift (simplified)
        k_eff = k
        
        amp = feff_amplitude(k_eff, Z)
        phase = feff_phase(k_eff, Z)
        
        chi += (N * s02 * amp / (k * R**2) *
                np.sin(2 * k * R + phase + 2 * k * dE0 * 0.01) *
                np.exp(-2 * sig2 * k**2) *
                np.exp(-2 * R / lam))
    
    return chi
