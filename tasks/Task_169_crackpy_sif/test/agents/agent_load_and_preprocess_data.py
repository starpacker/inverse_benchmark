import numpy as np

import matplotlib

matplotlib.use('Agg')

def williams_mode1_ux(r, theta, n, mu, kappa):
    """Mode I contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa + n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
          - (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val

def williams_mode1_uy(r, theta, n, mu, kappa):
    """Mode I contribution to u_y for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa - n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
          + (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val

def williams_mode2_ux(r, theta, n, mu, kappa):
    """Mode II contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa + n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
          - (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val

def williams_mode2_uy(r, theta, n, mu, kappa):
    """Mode II contribution to u_y for term index n."""
    rn = r ** (n / 2.0)
    val = -(kappa - n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
          + (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val

def load_and_preprocess_data(E, nu, K_I_true, K_II_true, N_terms, Nr, Ntheta, snr_db, seed_gt=42, seed_noise=123):
    """
    Generate synthetic DIC displacement data around a crack tip.
    
    Parameters
    ----------
    E : float
        Young's modulus [MPa]
    nu : float
        Poisson's ratio
    K_I_true : float
        Ground-truth Mode I SIF [MPa√m]
    K_II_true : float
        Ground-truth Mode II SIF [MPa√m]
    N_terms : int
        Number of Williams series terms
    Nr : int
        Number of radial grid points
    Ntheta : int
        Number of angular grid points
    snr_db : float
        Signal-to-noise ratio in dB
    seed_gt : int
        Random seed for ground truth higher-order terms
    seed_noise : int
        Random seed for noise generation
    
    Returns
    -------
    data_dict : dict
        Dictionary containing all data and parameters needed for inversion
    """
    # Material parameters
    mu = E / (2.0 * (1.0 + nu))
    kappa = (3.0 - nu) / (1.0 + nu)
    
    # Create polar grid around crack tip
    r_vals = np.linspace(1e-3, 10e-3, Nr)      # 1 mm to 10 mm (in metres)
    theta_vals = np.linspace(-np.pi * 0.95, np.pi * 0.95, Ntheta)
    R, THETA = np.meshgrid(r_vals, theta_vals)
    r_flat = R.ravel()
    theta_flat = THETA.ravel()
    
    # Ground-truth Williams coefficients
    coeffs_I_true = np.zeros(N_terms)
    coeffs_II_true = np.zeros(N_terms)
    coeffs_I_true[0] = K_I_true / np.sqrt(2.0 * np.pi)
    coeffs_II_true[0] = K_II_true / np.sqrt(2.0 * np.pi)
    
    # Add small higher-order terms for realism
    np.random.seed(seed_gt)
    for i in range(1, N_terms):
        coeffs_I_true[i] = np.random.uniform(-0.5, 0.5)
        coeffs_II_true[i] = np.random.uniform(-0.3, 0.3)
    
    # Generate clean displacement field using forward model
    ux_clean = np.zeros_like(r_flat)
    uy_clean = np.zeros_like(r_flat)
    for i in range(N_terms):
        n = i + 1
        ux_clean += coeffs_I_true[i] * williams_mode1_ux(r_flat, theta_flat, n, mu, kappa)
        uy_clean += coeffs_I_true[i] * williams_mode1_uy(r_flat, theta_flat, n, mu, kappa)
        ux_clean += coeffs_II_true[i] * williams_mode2_ux(r_flat, theta_flat, n, mu, kappa)
        uy_clean += coeffs_II_true[i] * williams_mode2_uy(r_flat, theta_flat, n, mu, kappa)
    
    # Add Gaussian noise
    signal_power_ux = np.mean(ux_clean ** 2)
    signal_power_uy = np.mean(uy_clean ** 2)
    noise_power_ux = signal_power_ux / (10 ** (snr_db / 10.0))
    noise_power_uy = signal_power_uy / (10 ** (snr_db / 10.0))
    
    np.random.seed(seed_noise)
    noise_ux = np.random.normal(0, np.sqrt(noise_power_ux), ux_clean.shape)
    noise_uy = np.random.normal(0, np.sqrt(noise_power_uy), uy_clean.shape)
    
    ux_noisy = ux_clean + noise_ux
    uy_noisy = uy_clean + noise_uy
    
    # Build design matrix for linear least-squares
    n_pts = len(r_flat)
    n_coeffs = 2 * N_terms
    M = np.zeros((2 * n_pts, n_coeffs))
    
    for i in range(N_terms):
        n = i + 1
        # Mode I columns
        M[:n_pts, i] = williams_mode1_ux(r_flat, theta_flat, n, mu, kappa)
        M[n_pts:, i] = williams_mode1_uy(r_flat, theta_flat, n, mu, kappa)
        # Mode II columns
        M[:n_pts, N_terms + i] = williams_mode2_ux(r_flat, theta_flat, n, mu, kappa)
        M[n_pts:, N_terms + i] = williams_mode2_uy(r_flat, theta_flat, n, mu, kappa)
    
    # Data vector
    d = np.concatenate([ux_noisy, uy_noisy])
    
    data_dict = {
        'r_flat': r_flat,
        'theta_flat': theta_flat,
        'ux_clean': ux_clean,
        'uy_clean': uy_clean,
        'ux_noisy': ux_noisy,
        'uy_noisy': uy_noisy,
        'design_matrix': M,
        'data_vector': d,
        'coeffs_I_true': coeffs_I_true,
        'coeffs_II_true': coeffs_II_true,
        'K_I_true': K_I_true,
        'K_II_true': K_II_true,
        'N_terms': N_terms,
        'E': E,
        'nu': nu,
        'mu': mu,
        'kappa': kappa,
        'snr_db': snr_db,
    }
    
    return data_dict
