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

def forward_operator(coeffs, r_flat, theta_flat, N_terms, mu, kappa):
    """
    Compute displacement field from Williams coefficients.
    
    Parameters
    ----------
    coeffs : ndarray
        Combined coefficient array [A_1, ..., A_N, B_1, ..., B_N]
    r_flat : ndarray
        Radial coordinates (flattened)
    theta_flat : ndarray
        Angular coordinates (flattened)
    N_terms : int
        Number of Williams series terms
    mu : float
        Shear modulus
    kappa : float
        Kolosov constant
    
    Returns
    -------
    y_pred : ndarray
        Predicted displacement vector [ux_1, ..., ux_n, uy_1, ..., uy_n]
    """
    coeffs_I = coeffs[:N_terms]
    coeffs_II = coeffs[N_terms:]
    
    ux = np.zeros_like(r_flat)
    uy = np.zeros_like(r_flat)
    
    for i in range(N_terms):
        n = i + 1
        ux += coeffs_I[i] * williams_mode1_ux(r_flat, theta_flat, n, mu, kappa)
        uy += coeffs_I[i] * williams_mode1_uy(r_flat, theta_flat, n, mu, kappa)
        ux += coeffs_II[i] * williams_mode2_ux(r_flat, theta_flat, n, mu, kappa)
        uy += coeffs_II[i] * williams_mode2_uy(r_flat, theta_flat, n, mu, kappa)
    
    y_pred = np.concatenate([ux, uy])
    return y_pred

def run_inversion(data_dict):
    """
    Solve the inverse problem using linear least-squares.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing design matrix, data vector, and parameters
    
    Returns
    -------
    result : dict
        Dictionary containing fitted coefficients, SIFs, and reconstructed displacements
    """
    M = data_dict['design_matrix']
    d = data_dict['data_vector']
    N_terms = data_dict['N_terms']
    r_flat = data_dict['r_flat']
    theta_flat = data_dict['theta_flat']
    mu = data_dict['mu']
    kappa = data_dict['kappa']
    
    # Ordinary least squares: c = (M^T M)^{-1} M^T d
    coeffs_fit, residuals, rank, sv = np.linalg.lstsq(M, d, rcond=None)
    
    A_fit = coeffs_fit[:N_terms]
    B_fit = coeffs_fit[N_terms:]
    
    K_I_fit = A_fit[0] * np.sqrt(2.0 * np.pi)
    K_II_fit = B_fit[0] * np.sqrt(2.0 * np.pi)
    
    # Compute reconstructed displacement using forward operator
    y_pred = forward_operator(coeffs_fit, r_flat, theta_flat, N_terms, mu, kappa)
    n_pts = len(r_flat)
    ux_fit = y_pred[:n_pts]
    uy_fit = y_pred[n_pts:]
    
    result = {
        'coeffs_fit': coeffs_fit,
        'A_fit': A_fit,
        'B_fit': B_fit,
        'K_I_fit': K_I_fit,
        'K_II_fit': K_II_fit,
        'ux_fit': ux_fit,
        'uy_fit': uy_fit,
        'residuals': residuals,
        'rank': rank,
        'singular_values': sv,
    }
    
    return result
