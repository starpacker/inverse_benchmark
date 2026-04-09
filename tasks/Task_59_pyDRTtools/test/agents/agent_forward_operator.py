import numpy as np

import matplotlib

matplotlib.use('Agg')

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
