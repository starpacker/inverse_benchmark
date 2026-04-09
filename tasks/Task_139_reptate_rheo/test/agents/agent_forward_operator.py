import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(omega, G0, tau_R, N_modes, eta_s=0.0):
    """
    Compute storage (G') and loss (G'') moduli using the Rouse model.

    The Rouse model describes unentangled polymer dynamics:
        G'(ω)  = G0 * Σ_{p=1}^{N} ω²τ_p² / (1 + ω²τ_p²)
        G''(ω) = G0 * Σ_{p=1}^{N} ωτ_p   / (1 + ω²τ_p²)  + ω η_s
    where τ_p = τ_R / p² are the Rouse relaxation times.

    Parameters
    ----------
    omega : ndarray
        Angular frequencies (rad/s).
    G0 : float
        Modulus prefactor nkT (Pa).
    tau_R : float
        Longest Rouse relaxation time (s).
    N_modes : int
        Number of Rouse modes to sum.
    eta_s : float
        Solvent viscosity contribution (Pa·s).

    Returns
    -------
    G_prime : ndarray
        Storage modulus G'(ω).
    G_double_prime : ndarray
        Loss modulus G''(ω).
    """
    omega = np.asarray(omega, dtype=np.float64)
    G_prime = np.zeros_like(omega)
    G_double_prime = np.zeros_like(omega)

    for p in range(1, N_modes + 1):
        tau_p = tau_R / p**2
        wt = omega * tau_p
        wt2 = wt * wt
        denom = 1.0 + wt2
        G_prime += G0 * wt2 / denom
        G_double_prime += G0 * wt / denom

    G_double_prime += omega * eta_s
    return G_prime, G_double_prime
