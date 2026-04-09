import numpy as np

import matplotlib

matplotlib.use("Agg")

def euler_bernoulli_element(L_e, EI, rhoA):
    """
    4×4 element stiffness and consistent mass matrices for an
    Euler-Bernoulli beam element of length L_e.
    """
    k_e = (EI / L_e ** 3) * np.array([
        [ 12,    6*L_e,  -12,    6*L_e],
        [  6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12,   -6*L_e,   12,   -6*L_e],
        [  6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])

    m_e = (rhoA * L_e / 420.0) * np.array([
        [156,    22*L_e,   54,   -13*L_e],
        [ 22*L_e,  4*L_e**2,  13*L_e, -3*L_e**2],
        [ 54,    13*L_e,  156,   -22*L_e],
        [-13*L_e, -3*L_e**2, -22*L_e,  4*L_e**2]
    ])
    return k_e, m_e
