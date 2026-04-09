import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def nmf_unmixing(Y, n_end, n_iter=500, rng=None):
    """
    Non-negative Matrix Factorisation for joint E,A estimation.
    """
    L, P = Y.shape
    R = n_end
    
    Y_pos = np.maximum(Y, 0)
    
    if rng is None:
        rng = np.random.default_rng(42)
    E = np.abs(rng.standard_normal((L, R))) + 0.1
    A = np.abs(rng.standard_normal((R, P))) + 0.1
    
    A /= A.sum(axis=0, keepdims=True)
    
    eps = 1e-10
    for it in range(n_iter):
        num_A = E.T @ Y_pos
        den_A = E.T @ E @ A + eps
        A *= (num_A / den_A)
        
        A = np.maximum(A, eps)
        A /= A.sum(axis=0, keepdims=True)
        
        num_E = Y_pos @ A.T
        den_E = E @ A @ A.T + eps
        E *= (num_E / den_E)
        E = np.maximum(E, eps)
        
        if (it + 1) % 100 == 0:
            err = np.linalg.norm(Y_pos - E @ A, 'fro') / np.linalg.norm(Y_pos, 'fro')
            print(f"    NMF iter {it+1}: rel_error={err:.6f}")
    
    return E, A
