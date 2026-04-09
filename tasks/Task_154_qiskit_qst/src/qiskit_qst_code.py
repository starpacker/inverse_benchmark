"""
qiskit_qst - Quantum State Tomography Inverse Problem
=====================================================
Task: Reconstruct density matrix from Pauli measurement statistics
Repo: https://github.com/qiskit-community/qiskit-experiments

Forward Problem: Known quantum state |ψ⟩ → Pauli-basis measurements → statistics
Inverse Problem: Measurement statistics → reconstruct density matrix ρ
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from itertools import product

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS_1Q = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def tensor(*mats):
    """Kronecker product of multiple matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


# ---------------------------------------------------------------------------
# State preparation helpers
# ---------------------------------------------------------------------------
def bell_state_density_matrix():
    """Create the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 density matrix."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    return rho


def random_pure_state_density_matrix(n_qubits):
    """Create a random pure state density matrix for n qubits."""
    dim = 2 ** n_qubits
    # Random state vector from Haar measure
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return rho


def ghz_state_density_matrix(n_qubits):
    """Create the GHZ state (|00...0⟩ + |11...1⟩)/√2 density matrix."""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    return rho


# ---------------------------------------------------------------------------
# Forward problem: simulate Pauli measurements
# ---------------------------------------------------------------------------
def generate_pauli_basis_labels(n_qubits):
    """Generate all n-qubit Pauli basis labels (excluding all-I for traceless)."""
    labels = []
    for combo in product(['I', 'X', 'Y', 'Z'], repeat=n_qubits):
        labels.append(''.join(combo))
    return labels


def pauli_operator(label):
    """Convert a Pauli label string to the matrix operator."""
    mats = [PAULIS_1Q[c] for c in label]
    return tensor(*mats)


def simulate_pauli_measurements(rho, n_qubits, n_shots=10000):
    """
    Forward problem: Given density matrix rho, simulate Pauli measurements.
    Returns dict {pauli_label: expectation_value}.

    For each Pauli operator P, ⟨P⟩ = Tr(ρ·P).
    We add shot noise: observed = ⟨P⟩ + noise ~ N(0, 1/√n_shots).
    """
    labels = generate_pauli_basis_labels(n_qubits)
    measurements = {}
    for label in labels:
        P = pauli_operator(label)
        # True expectation value
        exp_val = np.real(np.trace(rho @ P))
        # Add shot noise (statistical noise from finite measurements)
        noise = np.random.normal(0, 1.0 / np.sqrt(n_shots))
        measurements[label] = exp_val + noise
    return measurements


# ---------------------------------------------------------------------------
# Inverse problem: reconstruct density matrix
# ---------------------------------------------------------------------------
def linear_inversion_reconstruct(measurements, n_qubits):
    """
    Linear inversion QST:
    ρ = (1/2^n) Σ_P ⟨P⟩ P
    where the sum is over all n-qubit Pauli operators.
    """
    dim = 2 ** n_qubits
    rho_recon = np.zeros((dim, dim), dtype=complex)
    for label, exp_val in measurements.items():
        P = pauli_operator(label)
        rho_recon += exp_val * P
    rho_recon /= dim
    return rho_recon


def project_to_physical(rho):
    """
    Project a Hermitian matrix to the nearest valid density matrix.
    Ensures: Hermitian, positive semi-definite, trace = 1.
    """
    # Make Hermitian
    rho = (rho + rho.conj().T) / 2.0
    # Eigendecompose and clip negative eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    # Normalize trace to 1
    if np.sum(eigenvalues) > 0:
        eigenvalues /= np.sum(eigenvalues)
    rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    return rho_physical


def mle_reconstruct(measurements, n_qubits, max_iter=2000, tol=1e-10):
    """
    Maximum Likelihood Estimation (MLE) reconstruction using constrained
    least-squares optimization over the Cholesky parametrization.

    We parameterize ρ = T†T / Tr(T†T) where T is lower-triangular,
    ensuring ρ is always positive semi-definite with trace 1.

    Minimizes: Σ_P (⟨P⟩_obs - Tr(ρ·P))^2
    """
    from scipy.optimize import minimize

    dim = 2 ** n_qubits

    # Precompute Pauli operators and observed expectations
    pauli_ops = []
    exp_obs_list = []
    for label, exp_obs in measurements.items():
        pauli_ops.append(pauli_operator(label))
        exp_obs_list.append(exp_obs)
    pauli_ops = np.array(pauli_ops)   # (n_paulis, dim, dim)
    exp_obs_arr = np.array(exp_obs_list)

    def cholesky_to_rho(params):
        """Convert real parameter vector to density matrix via Cholesky."""
        # Build lower-triangular T from params
        T = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: real and positive
                    T[i, j] = params[idx]
                    idx += 1
                else:
                    # Off-diagonal: complex
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
    rho_init = linear_inversion_reconstruct(measurements, n_qubits)
    rho_init = project_to_physical(rho_init)

    # Cholesky decomposition of initial rho
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
    print(f"  MLE optimization: success={result.success}, cost={result.fun:.2e}, niter={result.nit}")

    return rho_mle


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def state_fidelity(rho_true, rho_recon):
    """
    Fidelity F(ρ, σ) = (Tr(√(√ρ·σ·√ρ)))^2
    For pure state ρ = |ψ⟩⟨ψ|: F = ⟨ψ|σ|ψ⟩
    """
    sqrt_true = matrix_sqrt(rho_true)
    product = sqrt_true @ rho_recon @ sqrt_true
    # Eigenvalues of product should be non-negative
    eigenvalues = np.linalg.eigvalsh(product)
    eigenvalues = np.maximum(eigenvalues, 0)
    fidelity = (np.sum(np.sqrt(eigenvalues))) ** 2
    return np.real(min(fidelity, 1.0))


def matrix_sqrt(M):
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


def trace_distance(rho_true, rho_recon):
    """
    Trace distance T(ρ, σ) = (1/2) Tr|ρ - σ|
    where |A| = √(A†A). For Hermitian A, |A| has eigenvalues |λ_i|.
    """
    diff = rho_true - rho_recon
    # diff is Hermitian, so eigenvalues are real
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def density_matrix_psnr(rho_true, rho_recon):
    """
    PSNR computed element-wise on the real and imaginary parts of density matrix.
    Treats the density matrix as a 2-channel image (real, imag).
    """
    true_real = np.real(rho_true)
    true_imag = np.imag(rho_true)
    recon_real = np.real(rho_recon)
    recon_imag = np.imag(rho_recon)

    # Combine real and imag as separate channels
    true_combined = np.concatenate([true_real.flatten(), true_imag.flatten()])
    recon_combined = np.concatenate([recon_real.flatten(), recon_imag.flatten()])

    mse = np.mean((true_combined - recon_combined) ** 2)
    if mse < 1e-15:
        return 60.0  # Essentially perfect reconstruction
    max_val = np.max(np.abs(true_combined)) if np.max(np.abs(true_combined)) > 0 else 1.0
    psnr = 10.0 * np.log10(max_val ** 2 / mse)
    return psnr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_density_matrices(rho_true, rho_recon_linear, rho_recon_mle, title_prefix, save_path):
    """
    Plot density matrices: true vs linear inversion vs MLE reconstruction.
    Shows real and imaginary parts as heatmaps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}: Quantum State Tomography', fontsize=16, fontweight='bold')

    matrices = [rho_true, rho_recon_linear, rho_recon_mle]
    titles = ['Ground Truth ρ', 'Linear Inversion', 'MLE Reconstruction']

    # Find global color range
    all_real = [np.real(m) for m in matrices]
    all_imag = [np.imag(m) for m in matrices]
    vmin_r = min(m.min() for m in all_real)
    vmax_r = max(m.max() for m in all_real)
    vmin_i = min(m.min() for m in all_imag)
    vmax_i = max(m.max() for m in all_imag)

    for col, (mat, title) in enumerate(zip(matrices, titles)):
        dim = mat.shape[0]
        # Real part
        im_r = axes[0, col].imshow(np.real(mat), cmap='RdBu_r', vmin=vmin_r, vmax=vmax_r,
                                    aspect='equal', interpolation='nearest')
        axes[0, col].set_title(f'{title}\n(Real part)', fontsize=12)
        axes[0, col].set_xlabel('Column')
        axes[0, col].set_ylabel('Row')
        plt.colorbar(im_r, ax=axes[0, col], shrink=0.8)

        # Add value annotations for small matrices
        if dim <= 4:
            for i in range(dim):
                for j in range(dim):
                    val = np.real(mat[i, j])
                    axes[0, col].text(j, i, f'{val:.3f}', ha='center', va='center',
                                       fontsize=8, color='black' if abs(val) < 0.3 else 'white')

        # Imaginary part
        im_i = axes[1, col].imshow(np.imag(mat), cmap='RdBu_r', vmin=vmin_i, vmax=vmax_i,
                                    aspect='equal', interpolation='nearest')
        axes[1, col].set_title(f'{title}\n(Imaginary part)', fontsize=12)
        axes[1, col].set_xlabel('Column')
        axes[1, col].set_ylabel('Row')
        plt.colorbar(im_i, ax=axes[1, col], shrink=0.8)

        if dim <= 4:
            for i in range(dim):
                for j in range(dim):
                    val = np.imag(mat[i, j])
                    axes[1, col].text(j, i, f'{val:.3f}', ha='center', va='center',
                                       fontsize=8, color='black' if abs(val) < 0.3 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")


# ---------------------------------------------------------------------------
# Qiskit-based QST (uses actual quantum circuits + AerSimulator)
# ---------------------------------------------------------------------------
def qiskit_qst_pipeline(n_qubits=2, n_shots=8192):
    """
    Full QST pipeline using Qiskit AerSimulator:
    1. Prepare a quantum state (Bell state for 2 qubits)
    2. Measure in all Pauli bases
    3. Reconstruct using linear inversion + MLE
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    print(f"\n{'='*60}")
    print(f"Qiskit QST Pipeline ({n_qubits} qubits, {n_shots} shots)")
    print(f"{'='*60}")

    sim = AerSimulator()

    # --- Step 1: Create the target state ---
    if n_qubits == 2:
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc_state = QuantumCircuit(2)
        qc_state.h(0)
        qc_state.cx(0, 1)
        state_name = "Bell State |Φ+⟩"
    else:
        # GHZ state
        qc_state = QuantumCircuit(n_qubits)
        qc_state.h(0)
        for i in range(n_qubits - 1):
            qc_state.cx(i, i + 1)
        state_name = f"GHZ State ({n_qubits} qubits)"

    print(f"  Target state: {state_name}")

    # Get true density matrix from statevector simulation
    qc_sv = qc_state.copy()
    qc_sv.save_statevector()
    result_sv = sim.run(qc_sv, shots=1).result()
    statevector = np.array(result_sv.get_statevector(qc_sv))
    rho_true = np.outer(statevector, statevector.conj())
    print(f"  True density matrix shape: {rho_true.shape}")
    print(f"  Tr(ρ_true) = {np.real(np.trace(rho_true)):.6f}")

    # --- Step 2: Forward problem - Pauli measurements ---
    print("\n  Running Pauli-basis measurements...")
    pauli_labels = generate_pauli_basis_labels(n_qubits)
    measurements = {}

    for label in pauli_labels:
        # Build measurement circuit
        qc_meas = qc_state.copy()

        # Apply basis rotation for each qubit
        for q, basis in enumerate(label):
            if basis == 'X':
                qc_meas.h(q)
            elif basis == 'Y':
                qc_meas.sdg(q)
                qc_meas.h(q)
            # Z and I basis: no rotation needed (measure in Z)

        qc_meas.measure_all()

        # Run on simulator
        result = sim.run(qc_meas, shots=n_shots).result()
        counts = result.get_counts(0)

        # Compute expectation value ⟨P⟩ from counts
        # For Pauli P = P_{q0} ⊗ P_{q1} ⊗ ..., the eigenvalue is the product
        # of individual eigenvalues. For qubit q with non-I Pauli, eigenvalue
        # is (-1)^bit. For I, eigenvalue is always +1 (identity).
        # Qiskit bitstring ordering: rightmost bit = qubit 0.
        non_identity_qubits = [q for q, basis in enumerate(label) if basis != 'I']

        exp_val = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')
            # Qiskit: bits[-1] = qubit 0, bits[-2] = qubit 1, ...
            parity = 1
            for q in non_identity_qubits:
                bit_idx = len(bits) - 1 - q  # reverse indexing
                if bits[bit_idx] == '1':
                    parity *= -1
            exp_val += parity * count / total

        measurements[label] = exp_val

    print(f"  Completed {len(measurements)} Pauli measurements")

    # --- Step 3: Inverse problem - Reconstruction ---
    print("\n  Reconstructing density matrix...")

    # Linear inversion
    rho_linear = linear_inversion_reconstruct(measurements, n_qubits)
    rho_linear = project_to_physical(rho_linear)
    print(f"  Linear inversion: Tr(ρ) = {np.real(np.trace(rho_linear)):.6f}")

    # MLE
    rho_mle = mle_reconstruct(measurements, n_qubits, max_iter=1000)
    print(f"  MLE reconstruction: Tr(ρ) = {np.real(np.trace(rho_mle)):.6f}")

    return rho_true, rho_linear, rho_mle, state_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    all_metrics = {}

    # -----------------------------------------------------------------------
    # Experiment 1: Bell State QST with Qiskit AerSimulator
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Bell State QST (Qiskit AerSimulator)")
    print("=" * 70)

    rho_true, rho_linear, rho_mle, state_name = qiskit_qst_pipeline(
        n_qubits=2, n_shots=8192
    )

    # Compute metrics - Linear Inversion
    fid_lin = state_fidelity(rho_true, rho_linear)
    td_lin = trace_distance(rho_true, rho_linear)
    psnr_lin = density_matrix_psnr(rho_true, rho_linear)
    print(f"\n  Linear Inversion Metrics:")
    print(f"    Fidelity:       {fid_lin:.6f}")
    print(f"    Trace Distance: {td_lin:.6f}")
    print(f"    PSNR:           {psnr_lin:.2f} dB")

    # Compute metrics - MLE
    fid_mle = state_fidelity(rho_true, rho_mle)
    td_mle = trace_distance(rho_true, rho_mle)
    psnr_mle = density_matrix_psnr(rho_true, rho_mle)
    print(f"\n  MLE Reconstruction Metrics:")
    print(f"    Fidelity:       {fid_mle:.6f}")
    print(f"    Trace Distance: {td_mle:.6f}")
    print(f"    PSNR:           {psnr_mle:.2f} dB")

    # Visualization
    plot_density_matrices(
        rho_true, rho_linear, rho_mle,
        "Bell State (2-qubit)",
        os.path.join(RESULTS_DIR, "reconstruction_result.png")
    )

    all_metrics['bell_state'] = {
        'state_name': state_name,
        'n_qubits': 2,
        'linear_inversion': {
            'fidelity': float(fid_lin),
            'trace_distance': float(td_lin),
            'psnr_dB': float(psnr_lin),
        },
        'mle': {
            'fidelity': float(fid_mle),
            'trace_distance': float(td_mle),
            'psnr_dB': float(psnr_mle),
        }
    }

    # -----------------------------------------------------------------------
    # Experiment 2: Random Pure State (analytical simulation)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Random Pure State (Analytical Simulation)")
    print("=" * 70)

    rho_true_rand = random_pure_state_density_matrix(2)
    print(f"  Random state density matrix shape: {rho_true_rand.shape}")

    measurements_rand = simulate_pauli_measurements(rho_true_rand, 2, n_shots=10000)
    rho_lin_rand = linear_inversion_reconstruct(measurements_rand, 2)
    rho_lin_rand = project_to_physical(rho_lin_rand)
    rho_mle_rand = mle_reconstruct(measurements_rand, 2, max_iter=1000)

    fid_lin_r = state_fidelity(rho_true_rand, rho_lin_rand)
    td_lin_r = trace_distance(rho_true_rand, rho_lin_rand)
    psnr_lin_r = density_matrix_psnr(rho_true_rand, rho_lin_rand)
    fid_mle_r = state_fidelity(rho_true_rand, rho_mle_rand)
    td_mle_r = trace_distance(rho_true_rand, rho_mle_rand)
    psnr_mle_r = density_matrix_psnr(rho_true_rand, rho_mle_rand)

    print(f"\n  Linear Inversion: Fidelity={fid_lin_r:.6f}, TD={td_lin_r:.6f}, PSNR={psnr_lin_r:.2f} dB")
    print(f"  MLE:              Fidelity={fid_mle_r:.6f}, TD={td_mle_r:.6f}, PSNR={psnr_mle_r:.2f} dB")

    all_metrics['random_state'] = {
        'state_name': 'Random Pure State (2-qubit)',
        'n_qubits': 2,
        'linear_inversion': {
            'fidelity': float(fid_lin_r),
            'trace_distance': float(td_lin_r),
            'psnr_dB': float(psnr_lin_r),
        },
        'mle': {
            'fidelity': float(fid_mle_r),
            'trace_distance': float(td_mle_r),
            'psnr_dB': float(psnr_mle_r),
        }
    }

    # -----------------------------------------------------------------------
    # Use best (MLE Bell state) as primary result
    # -----------------------------------------------------------------------
    best_fidelity = fid_mle
    best_td = td_mle
    best_psnr = psnr_mle

    # Summary metrics
    metrics = {
        'task': 'qiskit_qst',
        'task_number': 154,
        'description': 'Quantum State Tomography: reconstruct density matrix from Pauli measurement statistics',
        'method': 'Linear Inversion + Maximum Likelihood Estimation (MLE)',
        'primary_result': {
            'state': 'Bell State |Φ+⟩ (2-qubit)',
            'reconstruction_method': 'MLE',
            'fidelity': float(best_fidelity),
            'trace_distance': float(best_td),
            'psnr_dB': float(best_psnr),
        },
        'experiments': all_metrics,
    }

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics to {metrics_path}")

    # Save ground truth and reconstruction as numpy arrays
    gt_path = os.path.join(RESULTS_DIR, "ground_truth.npy")
    recon_path = os.path.join(RESULTS_DIR, "reconstruction.npy")
    np.save(gt_path, rho_true)
    np.save(recon_path, rho_mle)
    print(f"  Saved ground_truth.npy ({rho_true.shape})")
    print(f"  Saved reconstruction.npy ({rho_mle.shape})")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best Fidelity (MLE, Bell):  {best_fidelity:.6f}")
    print(f"  Trace Distance:             {best_td:.6f}")
    print(f"  PSNR:                       {best_psnr:.2f} dB")
    print(f"  Results saved to:           {RESULTS_DIR}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = main()
