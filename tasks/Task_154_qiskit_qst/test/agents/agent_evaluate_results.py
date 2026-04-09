import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from itertools import product

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
