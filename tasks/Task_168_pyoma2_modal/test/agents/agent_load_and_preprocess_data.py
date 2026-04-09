import os

import numpy as np

from scipy import linalg, signal

from scipy.linalg import expm

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(n_dof, masses, stiffnesses, damping_ratios_true, fs, T):
    """
    Build structural system matrices, compute ground truth modal parameters,
    and simulate ambient vibration response.
    
    Returns a dictionary containing all data needed for inversion.
    """
    dt = 1.0 / fs
    n_samples = int(T * fs)
    t = np.arange(n_samples) * dt
    
    # Mass matrix
    M = np.diag(masses)
    
    # Stiffness matrix (tridiagonal for chain system)
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        K[i, i] += stiffnesses[i]
        if i + 1 < n_dof:
            K[i, i] += stiffnesses[i + 1]
            K[i, i + 1] -= stiffnesses[i + 1]
            K[i + 1, i] -= stiffnesses[i + 1]
    
    # Analytical eigenvalue problem -> ground truth
    eigenvalues, eigenvectors = linalg.eigh(K, M)
    omega_n = np.sqrt(eigenvalues)
    freq_true = omega_n / (2 * np.pi)
    
    # Normalise mode shapes (max = 1)
    mode_shapes_true = eigenvectors.copy()
    for i in range(n_dof):
        mode_shapes_true[:, i] /= np.max(np.abs(mode_shapes_true[:, i]))
    
    # Construct Rayleigh damping matrix
    omega1, omega2 = omega_n[0], omega_n[1]
    zeta1, zeta2 = damping_ratios_true[0], damping_ratios_true[1]
    
    A_ray = np.array([[1 / (2 * omega1), omega1 / 2],
                      [1 / (2 * omega2), omega2 / 2]])
    b_ray = np.array([zeta1, zeta2])
    alpha_ray, beta_ray = np.linalg.solve(A_ray, b_ray)
    C = alpha_ray * M + beta_ray * K
    
    # Effective damping ratios under Rayleigh model
    damping_ratios_effective = alpha_ray / (2 * omega_n) + beta_ray * omega_n / 2
    
    # Simulate ambient vibration response (state-space integration)
    F = np.random.randn(n_samples, n_dof) * 5.0
    
    M_inv = np.linalg.inv(M)
    A_ss = np.zeros((2 * n_dof, 2 * n_dof))
    A_ss[:n_dof, n_dof:] = np.eye(n_dof)
    A_ss[n_dof:, :n_dof] = -M_inv @ K
    A_ss[n_dof:, n_dof:] = -M_inv @ C
    
    B_ss = np.zeros((2 * n_dof, n_dof))
    B_ss[n_dof:, :] = M_inv
    
    # Discrete-time state-space (ZOH)
    Ad = expm(A_ss * dt)
    Bd = np.linalg.solve(A_ss, (Ad - np.eye(2 * n_dof))) @ B_ss
    
    # Simulate
    state = np.zeros(2 * n_dof)
    accelerations = np.zeros((n_samples, n_dof))
    
    for i in range(n_samples):
        accelerations[i] = M_inv @ (F[i] - C @ state[n_dof:] - K @ state[:n_dof])
        state = Ad @ state + Bd @ F[i]
    
    # Add measurement noise (SNR ~ 30 dB)
    for ch in range(n_dof):
        sig_power = np.var(accelerations[:, ch])
        noise_power = sig_power / (10 ** (30.0 / 10.0))
        accelerations[:, ch] += np.random.randn(n_samples) * np.sqrt(noise_power)
    
    data = {
        "n_dof": n_dof,
        "fs": fs,
        "T": T,
        "dt": dt,
        "n_samples": n_samples,
        "t": t,
        "M": M,
        "K": K,
        "C": C,
        "accelerations": accelerations,
        "freq_true": freq_true,
        "omega_n": omega_n,
        "damping_ratios_effective": damping_ratios_effective,
        "mode_shapes_true": mode_shapes_true,
    }
    
    return data
