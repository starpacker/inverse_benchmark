import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.linalg import eigh, expm

def forward_operator(state, M, C, K, M_inv, dt):
    """
    Forward operator: Single time step of discrete state-space simulation.
    Given current state and force, compute next state.
    
    Parameters:
        state: Current state vector [displacement; velocity] (2*n_dof,)
        M, C, K: System matrices
        M_inv: Inverse of mass matrix
        dt: Time step
    
    Returns:
        A_d: Discrete state transition matrix
        B_d: Discrete input matrix
    """
    n_dof = M.shape[0]

    # Continuous state matrix
    A_c = np.zeros((2 * n_dof, 2 * n_dof))
    A_c[:n_dof, n_dof:] = np.eye(n_dof)
    A_c[n_dof:, :n_dof] = -M_inv @ K
    A_c[n_dof:, n_dof:] = -M_inv @ C

    # Input matrix
    B_c = np.zeros((2 * n_dof, n_dof))
    B_c[n_dof:, :] = M_inv

    # Discrete state-space via matrix exponential
    A_d = expm(A_c * dt)
    B_d = np.linalg.solve(A_c, (A_d - np.eye(2 * n_dof))) @ B_c

    return A_d, B_d
