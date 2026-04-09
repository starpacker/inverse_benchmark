import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import differential_evolution

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

def run_inversion(omega, G_prime_obs, G_double_prime_obs, N_modes=20):
    """
    Recover Rouse model parameters from noisy G'/G'' data.

    Uses differential evolution (global) followed by L-BFGS-B polish (local).
    The optimization is performed in log-parameter space with log-space residuals
    to handle data spanning multiple decades.

    Parameters
    ----------
    omega : ndarray
        Angular frequencies (rad/s).
    G_prime_obs : ndarray
        Observed storage modulus G'(ω).
    G_double_prime_obs : ndarray
        Observed loss modulus G''(ω).
    N_modes : int
        Number of Rouse modes to use in the model.

    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - 'G_prime_fit': ndarray, fitted storage modulus
        - 'G_double_prime_fit': ndarray, fitted loss modulus
        - 'fitted_params': dict, recovered parameters
    """
    def objective(params_vec):
        """Log-space least-squares objective (handles multi-decade data well)."""
        log_G0, log_tau_R, log_eta_s = params_vec
        G0 = 10.0 ** log_G0
        tau_R = 10.0 ** log_tau_R
        eta_s = 10.0 ** log_eta_s

        G_p, G_pp = forward_operator(omega, G0, tau_R, N_modes, eta_s)

        EPS = 1e-30
        res_p = np.log10(G_p + EPS) - np.log10(G_prime_obs + EPS)
        res_pp = np.log10(G_pp + EPS) - np.log10(G_double_prime_obs + EPS)

        return float(np.sum(res_p**2 + res_pp**2))

    bounds = [
        (3.0, 7.0),    # log10(G0)   : 1e3 → 1e7  Pa
        (-4.0, 1.0),   # log10(τ_R)  : 1e-4 → 10  s
        (-1.0, 4.0),   # log10(η_s)  : 0.1 → 1e4  Pa·s
    ]

    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=2000,
        tol=1e-12,
        polish=True,
        popsize=25,
    )

    G0_fit = 10.0 ** result.x[0]
    tau_R_fit = 10.0 ** result.x[1]
    eta_s_fit = 10.0 ** result.x[2]

    fitted_params = {
        'G0': float(G0_fit),
        'tau_R': float(tau_R_fit),
        'N_modes': int(N_modes),
        'eta_s': float(eta_s_fit),
    }

    G_prime_fit, G_double_prime_fit = forward_operator(
        omega, G0_fit, tau_R_fit, N_modes, eta_s_fit,
    )

    result_dict = {
        'G_prime_fit': G_prime_fit,
        'G_double_prime_fit': G_double_prime_fit,
        'fitted_params': fitted_params,
    }

    return result_dict
