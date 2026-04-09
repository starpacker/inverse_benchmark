import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import nnls

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def fcls(Y, E):
    """
    Fully Constrained Least Squares (FCLS) abundance estimation.
    """
    L, P = Y.shape
    R = E.shape[1]
    A = np.zeros((R, P))

    for p in range(P):
        a, residual = nnls(E, Y[:, p])
        a_sum = a.sum()
        if a_sum > 1e-12:
            a /= a_sum
        else:
            a = np.ones(R) / R
        A[:, p] = a

    return A
