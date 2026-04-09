import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

I = np.array([[1, 0], [0, 1]], dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS_1Q = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def tensor(*mats):
    """Kronecker product of multiple matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result

def pauli_operator(label):
    """Convert a Pauli label string to the matrix operator."""
    mats = [PAULIS_1Q[c] for c in label]
    return tensor(*mats)
