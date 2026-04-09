import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(modal_params, omega):
    """
    Forward operator: Reconstruct FRF from modal parameters.
    
    H_ij(omega) = sum_r (phi_ri * phi_rj) / (omega_r^2 - omega^2 + 2j*zeta_r*omega_r*omega)
    
    Args:
        modal_params: dict with 'freq_est', 'zeta_est', 'phi_est'
        omega: frequency array (rad/s)
    
    Returns:
        H_recon: reconstructed FRF (complex array)
    """
    freq_est = modal_params['freq_est']
    zeta_est = modal_params['zeta_est']
    phi_est = modal_params['phi_est']
    
    n_dof = phi_est.shape[0]
    n_modes = phi_est.shape[1]
    N = len(omega)
    
    H_recon = np.zeros((N, n_dof), dtype=complex)
    for r in range(n_modes):
        wr = freq_est[r]
        zr = zeta_est[r]
        phi_r = phi_est[:, r]
        for i in range(n_dof):
            num = phi_r[i] * phi_r[0]
            denom = wr**2 - omega**2 + 2j * zr * wr * omega
            H_recon[:, i] += num / denom
    
    return H_recon
