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

def load_and_preprocess_data(n_qubits=2, n_shots=8192, use_qiskit=True):
    """
    Load and preprocess data for quantum state tomography.
    
    For Qiskit mode: Creates a quantum circuit, runs it on AerSimulator,
    and collects Pauli measurement statistics.
    
    For analytical mode: Creates a density matrix directly and simulates
    measurements with shot noise.
    
    Returns:
        rho_true: The true density matrix (ground truth)
        measurements: Dictionary {pauli_label: expectation_value}
        n_qubits: Number of qubits
        state_name: Name of the quantum state
    """
    if use_qiskit:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        sim = AerSimulator()
        
        if n_qubits == 2:
            qc_state = QuantumCircuit(2)
            qc_state.h(0)
            qc_state.cx(0, 1)
            state_name = "Bell State |Φ+⟩"
        else:
            qc_state = QuantumCircuit(n_qubits)
            qc_state.h(0)
            for i in range(n_qubits - 1):
                qc_state.cx(i, i + 1)
            state_name = f"GHZ State ({n_qubits} qubits)"
        
        # Get true density matrix from statevector simulation
        qc_sv = qc_state.copy()
        qc_sv.save_statevector()
        result_sv = sim.run(qc_sv, shots=1).result()
        statevector = np.array(result_sv.get_statevector(qc_sv))
        rho_true = np.outer(statevector, statevector.conj())
        
        # Run Pauli-basis measurements
        pauli_labels = generate_pauli_basis_labels(n_qubits)
        measurements = {}
        
        for label in pauli_labels:
            qc_meas = qc_state.copy()
            
            for q, basis in enumerate(label):
                if basis == 'X':
                    qc_meas.h(q)
                elif basis == 'Y':
                    qc_meas.sdg(q)
                    qc_meas.h(q)
            
            qc_meas.measure_all()
            result = sim.run(qc_meas, shots=n_shots).result()
            counts = result.get_counts(0)
            
            non_identity_qubits = [q for q, basis in enumerate(label) if basis != 'I']
            
            exp_val = 0.0
            total = sum(counts.values())
            for bitstring, count in counts.items():
                bits = bitstring.replace(' ', '')
                parity = 1
                for q in non_identity_qubits:
                    bit_idx = len(bits) - 1 - q
                    if bits[bit_idx] == '1':
                        parity *= -1
                exp_val += parity * count / total
            
            measurements[label] = exp_val
    else:
        # Analytical simulation with random pure state
        dim = 2 ** n_qubits
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi /= np.linalg.norm(psi)
        rho_true = np.outer(psi, psi.conj())
        state_name = f"Random Pure State ({n_qubits}-qubit)"
        
        # Simulate Pauli measurements with shot noise
        labels = generate_pauli_basis_labels(n_qubits)
        measurements = {}
        for label in labels:
            P = pauli_operator(label)
            exp_val = np.real(np.trace(rho_true @ P))
            noise = np.random.normal(0, 1.0 / np.sqrt(n_shots))
            measurements[label] = exp_val + noise
    
    return rho_true, measurements, n_qubits, state_name
