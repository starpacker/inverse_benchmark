import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import eigh, expm

def load_and_preprocess_data(fs=200.0, duration=120.0, seed=42):
    """
    Build 5-DOF spring-mass-damper system and simulate vibration response.
    Returns all necessary data for the inverse problem.
    """
    n_dof = 5

    # Masses (kg)
    masses = np.array([2.0, 2.5, 2.0, 1.5, 2.0])
    M = np.diag(masses)

    # Stiffnesses (N/m) — chain topology
    k_vals = np.array([1000.0, 800.0, 1200.0, 900.0, 1100.0, 700.0])
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        K[i, i] = k_vals[i] + k_vals[i + 1]
        if i > 0:
            K[i, i - 1] = -k_vals[i]
            K[i - 1, i] = -k_vals[i]

    # Solve undamped eigenvalue problem for natural frequencies and mode shapes
    eigenvalues, eigenvectors = eigh(K, M)
    omega_n = np.sqrt(eigenvalues)
    freq_true = omega_n / (2 * np.pi)

    # Proportional modal damping
    zeta_target = np.array([0.01, 0.015, 0.02, 0.025, 0.03])
    Phi_n = eigenvectors.copy()
    for j in range(n_dof):
        Phi_n[:, j] /= np.sqrt(eigenvectors[:, j] @ M @ eigenvectors[:, j])

    modal_damping = np.diag(2.0 * zeta_target * omega_n)
    C = M @ Phi_n @ modal_damping @ Phi_n.T @ M

    zeta_true = zeta_target.copy()
    phi_true = eigenvectors.copy()

    # Normalize mode shapes
    for j in range(n_dof):
        phi_true[:, j] = phi_true[:, j] / np.max(np.abs(phi_true[:, j])) * np.sign(
            phi_true[np.argmax(np.abs(phi_true[:, j])), j]
        )

    # Simulate response
    np.random.seed(seed)
    dt = 1.0 / fs
    n_samples = int(duration * fs)
    t = np.arange(n_samples) * dt

    M_inv = np.linalg.inv(M)

    # Continuous state matrix
    A_c = np.zeros((2 * n_dof, 2 * n_dof))
    A_c[:n_dof, n_dof:] = np.eye(n_dof)
    A_c[n_dof:, :n_dof] = -M_inv @ K
    A_c[n_dof:, n_dof:] = -M_inv @ C

    # Input matrix
    B_c = np.zeros((2 * n_dof, n_dof))
    B_c[n_dof:, :] = M_inv

    # Discrete state-space
    A_d = expm(A_c * dt)
    B_d = np.linalg.solve(A_c, (A_d - np.eye(2 * n_dof))) @ B_c

    # White noise excitation
    force_amplitude = 10.0
    F = force_amplitude * np.random.randn(n_samples, n_dof)

    # Simulate
    x = np.zeros((n_samples, 2 * n_dof))
    for k in range(n_samples - 1):
        x[k + 1] = A_d @ x[k] + B_d @ F[k]

    disp = x[:, :n_dof]
    vel = x[:, n_dof:]

    # Compute accelerations
    acc = np.zeros((n_samples, n_dof))
    for k in range(n_samples):
        acc[k] = M_inv @ (F[k] - C @ vel[k] - K @ disp[k])

    data = {
        'M': M,
        'C': C,
        'K': K,
        'n_dof': n_dof,
        'freq_true': freq_true,
        'zeta_true': zeta_true,
        'phi_true': phi_true,
        't': t,
        'acc': acc,
        'disp': disp,
        'force': F,
        'fs': fs,
        'duration': duration
    }

    return data
