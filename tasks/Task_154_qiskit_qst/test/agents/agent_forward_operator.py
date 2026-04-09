import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from itertools import product

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

def generate_pauli_basis_labels(n_qubits):
    """Generate all n-qubit Pauli basis labels."""
    labels = []
    for combo in product(['I', 'X', 'Y', 'Z'], repeat=n_qubits):
        labels.append(''.join(combo))
    return labels

def pauli_operator(label):
    """Convert a Pauli label string to the matrix operator."""
    mats = [PAULIS_1Q[c] for c in label]
    return tensor(*mats)

def forward_operator(rho, n_qubits):
    """
    Forward operator: Given density matrix rho, compute expected Pauli measurements.
    
    For each Pauli operator P, computes ⟨P⟩ = Tr(ρ·P).
    
    Args:
        rho: Density matrix (2^n x 2^n complex array)
        n_qubits: Number of qubits
    
    Returns:
        measurements: Dictionary {pauli_label: expectation_value}
    """
    labels = generate_pauli_basis_labels(n_qubits)
    measurements = {}
    for label in labels:
        P = pauli_operator(label)
        exp_val = np.real(np.trace(rho @ P))
        measurements[label] = exp_val
    return measurements
