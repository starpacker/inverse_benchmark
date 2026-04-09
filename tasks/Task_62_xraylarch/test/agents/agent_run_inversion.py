import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import least_squares, differential_evolution

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

def run_inversion(k, chi_meas, s02, k_weight, seed):
    """
    Fit EXAFS data to recover shell parameters.
    
    Uses Differential Evolution + Levenberg-Marquardt refinement.
    
    Parameters
    ----------
    k : ndarray
        Photoelectron wavenumber array [Å^-1].
    chi_meas : ndarray
        Measured (noisy) EXAFS oscillation.
    s02 : float
        Amplitude reduction factor.
    k_weight : int
        k-weighting exponent for fitting.
    seed : int
        Random seed for DE optimization.
    
    Returns
    -------
    fit_shells : list of dict
        Fitted shell parameters.
    chi_fit : ndarray
        Fitted EXAFS oscillation.
    """
    def residual(params):
        shells = [
            {"N": params[0], "R": params[1], "sigma2": params[2],
             "dE0": params[3], "Z": 8, "label": "Fe-O"},
            {"N": params[4], "R": params[5], "sigma2": params[6],
             "dE0": params[7], "Z": 26, "label": "Fe-Fe"},
        ]
        chi_calc = forward_operator(shells, k, s02)
        return (chi_meas - chi_calc) * k**k_weight
    
    bounds_de = [
        (1, 12), (1.5, 2.5), (0.001, 0.02), (-5, 10),
        (0.5, 6), (2.5, 3.8), (0.001, 0.02), (-5, 10),
    ]
    
    print("[RECON] Stage 1 — Differential Evolution ...")
    result_de = differential_evolution(
        lambda p: np.sum(residual(p)**2), bounds_de,
        seed=seed, maxiter=150, tol=1e-5
    )
    print(f"[RECON]   χ² = {result_de.fun:.6f}")
    
    print("[RECON] Stage 2 — Levenberg-Marquardt ...")
    lb = [b[0] for b in bounds_de]
    ub = [b[1] for b in bounds_de]
    result = least_squares(residual, result_de.x, bounds=(lb, ub),
                           method='trf', ftol=1e-8, xtol=1e-8)
    print(f"[RECON]   cost = {result.cost:.6f}")
    
    p = result.x
    fit_shells = [
        {"N": float(p[0]), "R": float(p[1]), "sigma2": float(p[2]),
         "dE0": float(p[3]), "Z": 8, "label": "Fe-O"},
        {"N": float(p[4]), "R": float(p[5]), "sigma2": float(p[6]),
         "dE0": float(p[7]), "Z": 26, "label": "Fe-Fe"},
    ]
    chi_fit = forward_operator(fit_shells, k, s02)
    
    return fit_shells, chi_fit
