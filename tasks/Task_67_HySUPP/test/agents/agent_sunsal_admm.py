import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def sunsal_admm(Y, E, lam=0.01, n_iter=200, rho=1.0):
    """
    Sparse Unmixing by Variable Splitting and Augmented Lagrangian (SUnSAL).
    """
    L, P = Y.shape
    R = E.shape[1]

    EtE = E.T @ E
    EtY = E.T @ Y
    I_R = np.eye(R)
    inv_mat = np.linalg.inv(EtE + rho * I_R)

    A = np.linalg.lstsq(E, Y, rcond=None)[0]
    Z = A.copy()
    D = np.zeros_like(A)

    for it in range(n_iter):
        A = inv_mat @ (EtY + rho * (Z - D))

        V = A + D
        Z = np.sign(V) * np.maximum(np.abs(V) - lam / rho, 0)
        Z = np.maximum(Z, 0)
        for p in range(P):
            z = Z[:, p]
            s = z.sum()
            if s > 1e-12:
                z /= s
            else:
                z = np.ones(R) / R
            Z[:, p] = z

        D = D + A - Z

    return Z
