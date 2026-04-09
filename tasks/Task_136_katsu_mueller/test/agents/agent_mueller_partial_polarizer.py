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

def mueller_partial_polarizer(diattenuation=0.5, theta=0.0):
    """Partial polarizer with specified diattenuation D ∈ [0, 1]."""
    D = diattenuation
    q = 0.5 * (1 + D)
    r = 0.5 * (1 - D)
    a = q + r
    b = q - r
    c = 2 * np.sqrt(q * r)
    M = np.array([
        [a, b, 0, 0],
        [b, a, 0, 0],
        [0, 0, c, 0],
        [0, 0, 0, c],
    ])
    if theta != 0.0:
        R = mueller_rotation(theta)
        Rinv = mueller_rotation(-theta)
        M = R @ M @ Rinv
    return M
