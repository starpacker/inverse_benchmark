import os

import numpy as np

from scipy.linalg import expm

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(x, M, C, K, dt, n_samples, n_dof):
    """
    Forward model: Given modal parameters (frequencies, damping, mode shapes),
    simulate system response to white noise excitation.
    
    x: dict with keys 'frequencies', 'damping_ratios', 'mode_shapes'
    
    Returns predicted accelerations (n_samples, n_dof).
    """
    frequencies = x["frequencies"]
    damping_ratios = x["damping_ratios"]
    mode_shapes = x["mode_shapes"]
    
    n_modes = len(frequencies)
    
    # Generate white noise force
    np.random.seed(123)
    F = np.random.randn(n_samples, n_dof) * 5.0
    
    M_inv = np.linalg.inv(M)
    
    # State-space formulation
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
    accelerations_pred = np.zeros((n_samples, n_dof))
    
    for i in range(n_samples):
        accelerations_pred[i] = M_inv @ (F[i] - C @ state[n_dof:] - K @ state[:n_dof])
        state = Ad @ state + Bd @ F[i]
    
    return accelerations_pred
