import matplotlib

matplotlib.use('Agg')

import numpy as np

def mueller_rotation(theta):
    """Rotation matrix R(theta) for Mueller calculus (angle in radians)."""
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return np.array([
        [1,  0,   0,   0],
        [0,  c2,  s2,  0],
        [0, -s2,  c2,  0],
        [0,  0,   0,   1],
    ])

def mueller_linear_retarder(delta, theta=0.0):
    """Mueller matrix of a linear retarder with retardance *delta* at angle *theta*."""
    cd = np.cos(delta)
    sd = np.sin(delta)
    M_ret = np.array([
        [1, 0,   0,    0],
        [0, 1,   0,    0],
        [0, 0,   cd,   sd],
        [0, 0,  -sd,   cd],
    ])
    R = mueller_rotation(theta)
    Rinv = mueller_rotation(-theta)
    return R @ M_ret @ Rinv

def mueller_ideal_polarizer_h():
    """Ideal horizontal linear polarizer."""
    return 0.5 * np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)

def build_measurement_matrix(theta_g_list, theta_a_list, delta_g=np.pi/2, delta_a=np.pi/2):
    """
    Build the N×16 measurement (instrument) matrix **W** for a DRR polarimeter.

    Parameters
    ----------
    theta_g_list : array-like  – generator retarder angles (rad)
    theta_a_list : array-like  – analyser retarder angles (rad)
    delta_g      : float       – generator retarder retardance (default QWP)
    delta_a      : float       – analyser retarder retardance (default QWP)

    Returns
    -------
    W : ndarray, shape (N, 16)
    """
    N = len(theta_g_list)
    W = np.zeros((N, 16))

    P_h = mueller_ideal_polarizer_h()
    S0 = np.array([1.0, 0.0, 0.0, 0.0])

    for k in range(N):
        M_g = mueller_linear_retarder(delta_g, theta_g_list[k])
        M_psg = M_g @ P_h
        S_in = M_psg @ S0

        M_a = mueller_linear_retarder(delta_a, theta_a_list[k])
        M_psa = P_h @ M_a
        D_out = M_psa[0, :]

        W[k, :] = np.outer(D_out, S_in).ravel()

    return W
