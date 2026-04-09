import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(noise_level=0.05, seed=42):
    """
    Generate and preprocess synthetic rheology data with known ground-truth parameters.

    Parameters
    ----------
    noise_level : float
        Relative noise level for multiplicative Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'omega': ndarray, angular frequencies (rad/s)
        - 'G_prime_obs': ndarray, observed storage modulus with noise
        - 'G_double_prime_obs': ndarray, observed loss modulus with noise
        - 'G_prime_true': ndarray, true storage modulus (noise-free)
        - 'G_double_prime_true': ndarray, true loss modulus (noise-free)
        - 'true_params': dict, ground truth parameters
    """
    true_params = {
        'G0': 1.5e5,       # Pa  – modulus prefactor
        'tau_R': 0.01,     # s   – longest Rouse time
        'N_modes': 20,     # –   – number of Rouse modes
        'eta_s': 50.0,     # Pa·s – solvent viscosity
    }

    # Frequency sweep: 5 decades, 100 points (log-spaced)
    omega = np.logspace(-2, 4, 100)

    # Compute true moduli using forward operator
    G_prime_true, G_double_prime_true = forward_operator(
        omega,
        true_params['G0'],
        true_params['tau_R'],
        true_params['N_modes'],
        true_params['eta_s'],
    )

    # Multiplicative Gaussian noise
    rng = np.random.default_rng(seed)
    G_prime_obs = G_prime_true * (1.0 + noise_level * rng.standard_normal(len(omega)))
    G_double_prime_obs = G_double_prime_true * (1.0 + noise_level * rng.standard_normal(len(omega)))

    # Enforce positivity
    G_prime_obs = np.maximum(G_prime_obs, 1e-3)
    G_double_prime_obs = np.maximum(G_double_prime_obs, 1e-3)

    data_dict = {
        'omega': omega,
        'G_prime_obs': G_prime_obs,
        'G_double_prime_obs': G_double_prime_obs,
        'G_prime_true': G_prime_true,
        'G_double_prime_true': G_double_prime_true,
        'true_params': true_params,
    }

    return data_dict

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
