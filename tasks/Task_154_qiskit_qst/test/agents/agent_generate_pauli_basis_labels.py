import matplotlib

matplotlib.use('Agg')

import os

from itertools import product

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_pauli_basis_labels(n_qubits):
    """Generate all n-qubit Pauli basis labels."""
    labels = []
    for combo in product(['I', 'X', 'Y', 'Z'], repeat=n_qubits):
        labels.append(''.join(combo))
    return labels
