import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import target function
from agent_run_inversion import run_inversion

# ============== INJECT REFEREE CODE (Reference B) ==============
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def matrix_sqrt(M):
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

def evaluate_results(rho_true, rho_recon_linear, rho_recon_mle, state_name, save_dir=None):
    """
    Evaluate reconstruction quality and save results.
    
    Computes:
    - State fidelity: F(ρ, σ) = (Tr(√(√ρ·σ·√ρ)))^2
    - Trace distance: T(ρ, σ) = (1/2) Tr|ρ - σ|
    - PSNR: Peak Signal-to-Noise Ratio on density matrix elements
    
    Args:
        rho_true: True density matrix
        rho_recon_linear: Linear inversion reconstruction
        rho_recon_mle: MLE reconstruction
        state_name: Name of the quantum state
        save_dir: Directory to save results (optional)
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    
    def state_fidelity(rho_true, rho_recon):
        sqrt_true = matrix_sqrt(rho_true)
        product = sqrt_true @ rho_recon @ sqrt_true
        eigenvalues = np.linalg.eigvalsh(product)
        eigenvalues = np.maximum(eigenvalues, 0)
        fidelity = (np.sum(np.sqrt(eigenvalues))) ** 2
        return np.real(min(fidelity, 1.0))
    
    def trace_distance(rho_true, rho_recon):
        diff = rho_true - rho_recon
        eigenvalues = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigenvalues))
    
    def density_matrix_psnr(rho_true, rho_recon):
        true_real = np.real(rho_true)
        true_imag = np.imag(rho_true)
        recon_real = np.real(rho_recon)
        recon_imag = np.imag(rho_recon)
        
        true_combined = np.concatenate([true_real.flatten(), true_imag.flatten()])
        recon_combined = np.concatenate([recon_real.flatten(), recon_imag.flatten()])
        
        mse = np.mean((true_combined - recon_combined) ** 2)
        if mse < 1e-15:
            return 60.0
        max_val = np.max(np.abs(true_combined)) if np.max(np.abs(true_combined)) > 0 else 1.0
        psnr = 10.0 * np.log10(max_val ** 2 / mse)
        return psnr
    
    # Compute metrics for linear inversion
    fid_lin = state_fidelity(rho_true, rho_recon_linear)
    td_lin = trace_distance(rho_true, rho_recon_linear)
    psnr_lin = density_matrix_psnr(rho_true, rho_recon_linear)
    
    # Compute metrics for MLE
    fid_mle = state_fidelity(rho_true, rho_recon_mle)
    td_mle = trace_distance(rho_true, rho_recon_mle)
    psnr_mle = density_matrix_psnr(rho_true, rho_recon_mle)
    
    print(f"\n  Linear Inversion Metrics:")
    print(f"    Fidelity:       {fid_lin:.6f}")
    print(f"    Trace Distance: {td_lin:.6f}")
    print(f"    PSNR:           {psnr_lin:.2f} dB")
    
    print(f"\n  MLE Reconstruction Metrics:")
    print(f"    Fidelity:       {fid_mle:.6f}")
    print(f"    Trace Distance: {td_mle:.6f}")
    print(f"    PSNR:           {psnr_mle:.2f} dB")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{state_name}: Quantum State Tomography', fontsize=16, fontweight='bold')
    
    matrices = [rho_true, rho_recon_linear, rho_recon_mle]
    titles = ['Ground Truth ρ', 'Linear Inversion', 'MLE Reconstruction']
    
    all_real = [np.real(m) for m in matrices]
    all_imag = [np.imag(m) for m in matrices]
    vmin_r = min(m.min() for m in all_real)
    vmax_r = max(m.max() for m in all_real)
    vmin_i = min(m.min() for m in all_imag)
    vmax_i = max(m.max() for m in all_imag)
    
    for col, (mat, title) in enumerate(zip(matrices, titles)):
        dim = mat.shape[0]
        im_r = axes[0, col].imshow(np.real(mat), cmap='RdBu_r', vmin=vmin_r, vmax=vmax_r,
                                    aspect='equal', interpolation='nearest')
        axes[0, col].set_title(f'{title}\n(Real part)', fontsize=12)
        axes[0, col].set_xlabel('Column')
        axes[0, col].set_ylabel('Row')
        plt.colorbar(im_r, ax=axes[0, col], shrink=0.8)
        
        if dim <= 4:
            for i in range(dim):
                for j in range(dim):
                    val = np.real(mat[i, j])
                    axes[0, col].text(j, i, f'{val:.3f}', ha='center', va='center',
                                       fontsize=8, color='black' if abs(val) < 0.3 else 'white')
        
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
    save_path = os.path.join(save_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")
    
    # Compile metrics
    metrics = {
        'task': 'qiskit_qst',
        'description': 'Quantum State Tomography: reconstruct density matrix from Pauli measurement statistics',
        'method': 'Linear Inversion + Maximum Likelihood Estimation (MLE)',
        'state_name': state_name,
        'linear_inversion': {
            'fidelity': float(fid_lin),
            'trace_distance': float(td_lin),
            'psnr_dB': float(psnr_lin),
        },
        'mle': {
            'fidelity': float(fid_mle),
            'trace_distance': float(td_mle),
            'psnr_dB': float(psnr_mle),
        },
        'primary_result': {
            'reconstruction_method': 'MLE',
            'fidelity': float(fid_mle),
            'trace_distance': float(td_mle),
            'psnr_dB': float(psnr_mle),
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    # Save ground truth and reconstruction as numpy arrays
    gt_path = os.path.join(save_dir, "ground_truth.npy")
    recon_path = os.path.join(save_dir, "reconstruction.npy")
    np.save(gt_path, rho_true)
    np.save(recon_path, rho_recon_mle)
    print(f"  Saved ground_truth.npy ({rho_true.shape})")
    print(f"  Saved reconstruction.npy ({rho_recon_mle.shape})")
    
    return metrics

# ============== END REFEREE CODE ==============


def main():
    """Main test function for run_inversion."""
    
    # Data paths provided
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # We should have at least one outer file
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    outer_path = outer_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args count: {len(args)}")
    print(f"Kwargs: {list(kwargs.keys())}")
    
    # Run the agent's run_inversion
    print("\n" + "="*60)
    print("Running Agent's run_inversion...")
    print("="*60)
    
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent's run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner files (chained execution)
    if inner_files:
        print("\nDetected chained execution pattern (closure/factory)")
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        if callable(agent_output):
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            final_result = agent_output
    else:
        # Direct execution
        print("\nDirect execution pattern")
        final_result = agent_output
        std_result = std_output
    
    print("\n" + "="*60)
    print("Analyzing Results...")
    print("="*60)
    
    # run_inversion returns a tuple: (rho_recon, info)
    # We need to compare the reconstructed density matrices
    
    agent_rho, agent_info = final_result
    std_rho, std_info = std_result
    
    print(f"\nAgent reconstruction shape: {agent_rho.shape}")
    print(f"Agent info: {agent_info}")
    print(f"\nStandard reconstruction shape: {std_rho.shape}")
    print(f"Standard info: {std_info}")
    
    # For evaluation, we need the ground truth density matrix
    # The evaluate_results function expects: rho_true, rho_recon_linear, rho_recon_mle
    # Since we don't have the true rho directly, we'll use the standard output as reference
    # and compare agent's output against it
    
    # Extract measurements from args to potentially reconstruct linear inversion
    measurements = args[0] if args else kwargs.get('measurements', {})
    n_qubits = args[1] if len(args) > 1 else kwargs.get('n_qubits', 1)
    
    # We'll use std_rho as our "ground truth" for comparison purposes
    # And run both linear and MLE with agent's function
    
    print("\n" + "="*60)
    print("Running evaluation comparison...")
    print("="*60)
    
    # Get linear inversion result from agent
    try:
        agent_linear, _ = run_inversion(measurements, n_qubits, method='linear')
    except Exception as e:
        print(f"Warning: Could not get linear inversion: {e}")
        agent_linear = agent_rho  # fallback
    
    # Use standard output as reference (ground truth proxy)
    # The standard output is considered the "correct" reconstruction
    rho_reference = std_rho
    
    # Calculate direct comparison metrics
    def state_fidelity(rho1, rho2):
        """Calculate state fidelity between two density matrices."""
        eigenvalues1, eigenvectors1 = np.linalg.eigh(rho1)
        eigenvalues1 = np.maximum(eigenvalues1, 0)
        sqrt_eigenvalues1 = np.sqrt(eigenvalues1)
        sqrt_rho1 = eigenvectors1 @ np.diag(sqrt_eigenvalues1) @ eigenvectors1.conj().T
        
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        eigenvalues = np.linalg.eigvalsh(product)
        eigenvalues = np.maximum(eigenvalues, 0)
        fidelity = (np.sum(np.sqrt(eigenvalues))) ** 2
        return np.real(min(fidelity, 1.0))
    
    def trace_distance(rho1, rho2):
        """Calculate trace distance between two density matrices."""
        diff = rho1 - rho2
        eigenvalues = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigenvalues))
    
    # Compare agent output directly to standard output
    fidelity_agent_vs_std = state_fidelity(agent_rho, std_rho)
    trace_dist_agent_vs_std = trace_distance(agent_rho, std_rho)
    
    print(f"\n  Direct Comparison (Agent vs Standard):")
    print(f"    Fidelity:       {fidelity_agent_vs_std:.6f}")
    print(f"    Trace Distance: {trace_dist_agent_vs_std:.6f}")
    
    # Check if agent output is valid density matrix
    agent_trace = np.trace(agent_rho)
    agent_hermitian_error = np.linalg.norm(agent_rho - agent_rho.conj().T)
    eigenvalues_agent = np.linalg.eigvalsh(agent_rho)
    
    print(f"\n  Agent Output Validity Check:")
    print(f"    Trace: {np.real(agent_trace):.6f} (should be 1.0)")
    print(f"    Hermitian error: {agent_hermitian_error:.2e}")
    print(f"    Min eigenvalue: {np.min(eigenvalues_agent):.6e} (should be >= 0)")
    
    # Check standard output validity
    std_trace = np.trace(std_rho)
    std_hermitian_error = np.linalg.norm(std_rho - std_rho.conj().T)
    eigenvalues_std = np.linalg.eigvalsh(std_rho)
    
    print(f"\n  Standard Output Validity Check:")
    print(f"    Trace: {np.real(std_trace):.6f}")
    print(f"    Hermitian error: {std_hermitian_error:.2e}")
    print(f"    Min eigenvalue: {np.min(eigenvalues_std):.6e}")
    
    # Final evaluation - use full evaluate_results with std_rho as "ground truth"
    print("\n" + "="*60)
    print("Full Evaluation (using Standard as Ground Truth proxy)...")
    print("="*60)
    
    try:
        # For this evaluation, we treat std_rho as the "true" state
        # and evaluate both linear and MLE reconstructions from agent
        metrics_agent = evaluate_results(
            rho_true=std_rho,
            rho_recon_linear=agent_linear,
            rho_recon_mle=agent_rho,
            state_name="Agent_Reconstruction",
            save_dir=RESULTS_DIR
        )
        
        score_agent_fidelity = metrics_agent['mle']['fidelity']
        score_agent_psnr = metrics_agent['mle']['psnr_dB']
        
    except Exception as e:
        print(f"Warning during evaluation: {e}")
        traceback.print_exc()
        score_agent_fidelity = fidelity_agent_vs_std
        score_agent_psnr = 0.0
    
    # Performance check
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    print(f"\nAgent vs Standard Fidelity: {fidelity_agent_vs_std:.6f}")
    print(f"Agent vs Standard Trace Distance: {trace_dist_agent_vs_std:.6f}")
    
    # Success criteria:
    # - Fidelity should be very high (close to 1.0) since we're comparing reconstruction methods
    # - Trace distance should be very low (close to 0)
    
    FIDELITY_THRESHOLD = 0.95  # Agent should achieve at least 95% fidelity with standard
    TRACE_DIST_THRESHOLD = 0.1  # Trace distance should be below 0.1
    
    success = True
    
    if fidelity_agent_vs_std < FIDELITY_THRESHOLD:
        print(f"\n❌ FAIL: Fidelity {fidelity_agent_vs_std:.6f} < threshold {FIDELITY_THRESHOLD}")
        success = False
    else:
        print(f"\n✓ PASS: Fidelity {fidelity_agent_vs_std:.6f} >= threshold {FIDELITY_THRESHOLD}")
    
    if trace_dist_agent_vs_std > TRACE_DIST_THRESHOLD:
        print(f"❌ FAIL: Trace distance {trace_dist_agent_vs_std:.6f} > threshold {TRACE_DIST_THRESHOLD}")
        success = False
    else:
        print(f"✓ PASS: Trace distance {trace_dist_agent_vs_std:.6f} <= threshold {TRACE_DIST_THRESHOLD}")
    
    # Check density matrix validity
    if np.abs(np.real(agent_trace) - 1.0) > 0.01:
        print(f"❌ FAIL: Trace {np.real(agent_trace):.6f} is not close to 1.0")
        success = False
    
    if np.min(eigenvalues_agent) < -1e-6:
        print(f"❌ FAIL: Negative eigenvalue {np.min(eigenvalues_agent):.6e} (not positive semi-definite)")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Agent's run_inversion performs acceptably!")
        sys.exit(0)
    else:
        print("TEST FAILED: Agent's run_inversion has performance issues!")
        sys.exit(1)


if __name__ == "__main__":
    main()