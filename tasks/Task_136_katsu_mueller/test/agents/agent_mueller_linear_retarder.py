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
