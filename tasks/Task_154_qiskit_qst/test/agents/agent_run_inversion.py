import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import minimize

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

def project_to_physical(rho):
    """
    Project a Hermitian matrix to the nearest valid density matrix.
    Ensures: Hermitian, positive semi-definite, trace = 1.
    """
    rho = (rho + rho.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    if np.sum(eigenvalues) > 0:
        eigenvalues /= np.sum(eigenvalues)
    rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    return rho_physical

def run_inversion(measurements, n_qubits, method='mle', max_iter=2000, tol=1e-10):
    """
    Inverse problem: Reconstruct density matrix from Pauli measurement statistics.
    
    Supports two methods:
    - 'linear': Linear inversion QST: ρ = (1/2^n) Σ_P ⟨P⟩ P
    - 'mle': Maximum Likelihood Estimation using Cholesky parametrization
    
    Args:
        measurements: Dictionary {pauli_label: expectation_value}
        n_qubits: Number of qubits
        method: 'linear' or 'mle'
        max_iter: Maximum iterations for MLE optimization
        tol: Convergence tolerance for MLE
    
    Returns:
        rho_recon: Reconstructed density matrix
        info: Dictionary with reconstruction information
    """
    dim = 2 ** n_qubits
    
    # Linear inversion reconstruction
    rho_linear = np.zeros((dim, dim), dtype=complex)
    for label, exp_val in measurements.items():
        P = pauli_operator(label)
        rho_linear += exp_val * P
    rho_linear /= dim
    rho_linear = project_to_physical(rho_linear)
    
    if method == 'linear':
        return rho_linear, {'method': 'linear_inversion', 'success': True}
    
    # MLE reconstruction
    pauli_ops = []
    exp_obs_list = []
    for label, exp_obs in measurements.items():
        pauli_ops.append(pauli_operator(label))
        exp_obs_list.append(exp_obs)
    pauli_ops = np.array(pauli_ops)
    exp_obs_arr = np.array(exp_obs_list)
    
    def cholesky_to_rho(params):
        """Convert real parameter vector to density matrix via Cholesky."""
        T = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    T[i, j] = params[idx]
                    idx += 1
                else:
                    T[i, j] = params[idx] + 1j * params[idx + 1]
                    idx += 2
        rho = T.conj().T @ T
        tr = np.trace(rho)
        if np.abs(tr) > 1e-15:
            rho /= tr
        return rho
    
    def cost_function(params):
        """Least-squares cost: sum of (observed - model)^2."""
        rho = cholesky_to_rho(params)
        cost = 0.0
        for k in range(len(pauli_ops)):
            exp_model = np.real(np.trace(rho @ pauli_ops[k]))
            cost += (exp_obs_arr[k] - exp_model) ** 2
        return cost
    
    # Initialize from linear inversion
    rho_init = rho_linear
    
    try:
        T_init = np.linalg.cholesky(rho_init + 1e-10 * np.eye(dim))
    except np.linalg.LinAlgError:
        T_init = np.eye(dim) / np.sqrt(dim)
    
    # Extract parameters from T_init
    params_init = []
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                params_init.append(np.real(T_init[i, j]))
            else:
                params_init.append(np.real(T_init[i, j]))
                params_init.append(np.imag(T_init[i, j]))
    params_init = np.array(params_init)
    
    # Optimize
    result = minimize(cost_function, params_init, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol})
    
    rho_mle = cholesky_to_rho(result.x)
    
    info = {
        'method': 'mle',
        'success': result.success,
        'cost': float(result.fun),
        'niter': result.nit
    }
    
    print(f"  MLE optimization: success={result.success}, cost={result.fun:.2e}, niter={result.nit}")
    
    return rho_mle, info
