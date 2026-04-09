import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Import target function
from agent_run_inversion import run_inversion

# Inject the evaluate_results function (Reference B)
def evaluate_results(obj_true, obj_recon, support, intensity, errors, results_dir):
    """
    Evaluate phase retrieval quality, align phases, compute metrics, and visualize.
    
    Args:
        obj_true: ground truth 3D complex object
        obj_recon: reconstructed 3D complex object
        support: 3D binary support mask
        intensity: measured diffraction intensity
        errors: convergence history
        results_dir: directory to save results
        
    Returns:
        metrics: dictionary of evaluation metrics
        obj_aligned: phase-aligned reconstruction
    """
    # Phase alignment - resolve twin-image ambiguity and global phase offset
    candidates = [obj_recon, np.conj(obj_recon[::-1, ::-1, ::-1])]
    
    best_cc = -1
    best_obj = None
    
    for cand in candidates:
        mask = support > 0.5
        if np.sum(mask) == 0:
            continue
        
        # Compute optimal phase offset
        cross = np.sum(cand[mask] * np.conj(obj_true[mask]))
        phase_offset = np.angle(cross)
        cand_aligned = cand * np.exp(-1j * phase_offset)
        
        # Compute correlation
        cc = np.abs(np.sum(cand_aligned[mask] * np.conj(obj_true[mask]))) / \
             (np.sqrt(np.sum(np.abs(cand_aligned[mask])**2) * np.sum(np.abs(obj_true[mask])**2)))
        
        if cc > best_cc:
            best_cc = cc
            best_obj = cand_aligned
    
    obj_aligned = best_obj
    print(f"[POST] Alignment CC = {best_cc:.6f}")
    
    # Compute metrics
    mask = support > 0.5
    
    # Amplitude metrics
    amp_true = np.abs(obj_true[mask])
    amp_recon = np.abs(obj_aligned[mask])
    
    amp_mse = np.mean((amp_true - amp_recon)**2)
    amp_range = amp_true.max() - amp_true.min()
    psnr_amp = 10 * np.log10(amp_range**2 / amp_mse) if amp_mse > 0 else float('inf')
    
    cc_amp = np.corrcoef(amp_true, amp_recon)[0, 1]
    
    # Phase metrics (within support)
    phase_true = np.angle(obj_true[mask])
    phase_recon = np.angle(obj_aligned[mask])
    
    # Phase difference (wrapped)
    phase_diff = np.angle(np.exp(1j * (phase_true - phase_recon)))
    phase_rmse = np.sqrt(np.mean(phase_diff**2))
    
    # Complex-valued correlation
    cc_complex = np.abs(np.sum(obj_aligned[mask] * np.conj(obj_true[mask]))) / \
                 np.sqrt(np.sum(np.abs(obj_aligned[mask])**2) * np.sum(np.abs(obj_true[mask])**2))
    
    # R-factor (crystallographic)
    ft_true = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_true))))
    ft_recon = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_aligned))))
    r_factor = np.sum(np.abs(ft_true - ft_recon)) / np.sum(ft_true)
    
    # Amplitude PSNR for the full 3D volume
    amp_full_true = np.abs(obj_true)
    amp_full_recon = np.abs(obj_aligned)
    mse_full = np.mean((amp_full_true - amp_full_recon)**2)
    range_full = amp_full_true.max() - amp_full_true.min()
    psnr_full = 10 * np.log10(range_full**2 / mse_full) if mse_full > 0 else float('inf')
    
    metrics = {
        'psnr_amplitude_support': float(psnr_amp),
        'psnr_amplitude_full': float(psnr_full),
        'cc_amplitude': float(cc_amp),
        'cc_complex': float(cc_complex),
        'phase_rmse_rad': float(phase_rmse),
        'r_factor': float(r_factor),
    }
    
    for k, v in metrics.items():
        print(f"[EVAL] {k} = {v:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "input.npy"), intensity)
    np.save(os.path.join(results_dir, "ground_truth.npy"), obj_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), obj_aligned)
    print(f"[SAVE] Input shape: {intensity.shape}")
    print(f"[SAVE] GT shape: {obj_true.shape}")
    print(f"[SAVE] Recon shape: {obj_aligned.shape}")
    
    # Visualization
    N = obj_true.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    mid = N // 2
    
    # (a) True amplitude - central slice
    ax = axes[0, 0]
    im = ax.imshow(np.abs(obj_true[:, :, mid]), cmap='hot', origin='lower')
    ax.set_title('True Amplitude (z-mid)')
    plt.colorbar(im, ax=ax)
    
    # (b) True phase - central slice
    ax = axes[0, 1]
    phase_true_slice = np.angle(obj_true[:, :, mid])
    phase_true_masked = np.where(support[:, :, mid] > 0.5, phase_true_slice, np.nan)
    im2 = ax.imshow(phase_true_masked, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title('True Phase (z-mid)')
    plt.colorbar(im2, ax=ax, label='rad')
    
    # (c) Diffraction intensity - central slice (log scale)
    ax = axes[0, 2]
    im3 = ax.imshow(np.log10(intensity[:, :, mid] + 1), cmap='viridis', origin='lower')
    ax.set_title('log₁₀(Intensity+1) (z-mid)')
    plt.colorbar(im3, ax=ax)
    
    # (d) Reconstructed amplitude
    ax = axes[0, 3]
    im4 = ax.imshow(np.abs(obj_aligned[:, :, mid]), cmap='hot', origin='lower')
    ax.set_title('Recon Amplitude (z-mid)')
    plt.colorbar(im4, ax=ax)
    
    # (e) Reconstructed phase
    ax = axes[1, 0]
    phase_recon_slice = np.angle(obj_aligned[:, :, mid])
    phase_recon_masked = np.where(support[:, :, mid] > 0.5, phase_recon_slice, np.nan)
    im5 = ax.imshow(phase_recon_masked, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Recon Phase (z-mid)')
    plt.colorbar(im5, ax=ax, label='rad')
    
    # (f) Phase error
    ax = axes[1, 1]
    phase_diff_slice = np.angle(np.exp(1j * (phase_true_slice - phase_recon_slice)))
    phase_diff_masked = np.where(support[:, :, mid] > 0.5, phase_diff_slice, np.nan)
    im6 = ax.imshow(phase_diff_masked, cmap='seismic', origin='lower', vmin=-0.5, vmax=0.5)
    ax.set_title('Phase Error (z-mid)')
    plt.colorbar(im6, ax=ax, label='rad')
    
    # (g) Convergence
    ax = axes[1, 2]
    ax.semilogy(errors)
    ax.set_xlabel('Iteration checkpoint')
    ax.set_ylabel('R-factor²')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)
    
    # (h) Amplitude scatter
    ax = axes[1, 3]
    ax.scatter(np.abs(obj_true[mask]), np.abs(obj_aligned[mask]),
               s=1, alpha=0.3, c='steelblue')
    lim = max(np.abs(obj_true[mask]).max(), np.abs(obj_aligned[mask]).max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', lw=2)
    ax.set_xlabel('True |ρ|')
    ax.set_ylabel('Recon |ρ|')
    ax.set_title(f'Amplitude (CC={metrics["cc_amplitude"]:.4f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"bcdi — Bragg CDI Phase Retrieval (HIO+ER)\n"
        f"PSNR={metrics['psnr_amplitude_full']:.2f} dB | "
        f"CC_complex={metrics['cc_complex']:.4f} | "
        f"Phase RMSE={metrics['phase_rmse_rad']:.4f} rad | "
        f"R-factor={metrics['r_factor']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")
    
    return metrics, obj_aligned


def main():
    # Data paths
    data_paths = ['/data/yjh/bcdi_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer data
        print("\n[LOAD] Loading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Function: {outer_data.get('func_name', 'unknown')}")
        print(f"[INFO] Args count: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        # Extract inputs for evaluation
        # Based on function signature: run_inversion(intensity, support, n_hio=200, n_er=50, n_cycles=5, beta=0.9)
        intensity = args[0] if len(args) > 0 else kwargs.get('intensity')
        support = args[1] if len(args) > 1 else kwargs.get('support')
        
        print(f"[INFO] Intensity shape: {intensity.shape}")
        print(f"[INFO] Support shape: {support.shape}")
        
        # Set random seed for reproducibility in comparison
        np.random.seed(42)
        
        # Run the agent's implementation
        print("\n[RUN] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is a chained execution
        if inner_data_paths:
            print("\n[CHAIN] Detected chained execution pattern...")
            # Handle inner function call
            inner_data_path = inner_data_paths[0]
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Call the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Unpack results (obj_recon, errors)
        if isinstance(final_result, tuple) and len(final_result) == 2:
            agent_obj_recon, agent_errors = final_result
        else:
            print("[ERROR] Unexpected output format from agent")
            sys.exit(1)
        
        if isinstance(std_result, tuple) and len(std_result) == 2:
            std_obj_recon, std_errors = std_result
        else:
            print("[ERROR] Unexpected output format from standard")
            sys.exit(1)
        
        print(f"\n[INFO] Agent reconstruction shape: {agent_obj_recon.shape}")
        print(f"[INFO] Standard reconstruction shape: {std_obj_recon.shape}")
        print(f"[INFO] Agent errors count: {len(agent_errors)}")
        print(f"[INFO] Standard errors count: {len(std_errors)}")
        
        # We need ground truth for evaluation
        # For phase retrieval, the "ground truth" is typically the original object
        # Since we don't have direct access, we'll use the standard result as reference
        # and compare reconstruction quality metrics
        
        # Create results directories
        agent_results_dir = "./agent_results"
        std_results_dir = "./std_results"
        
        # For evaluation, we need obj_true. Since this is phase retrieval,
        # we'll use the standard reconstruction as pseudo ground truth for comparison
        # This tests if the agent achieves similar quality
        
        print("\n[EVAL] Evaluating agent's reconstruction...")
        # Use standard result as reference
        agent_metrics, agent_aligned = evaluate_results(
            obj_true=std_obj_recon,  # Use standard as reference
            obj_recon=agent_obj_recon,
            support=support,
            intensity=intensity,
            errors=agent_errors,
            results_dir=agent_results_dir
        )
        
        print("\n[EVAL] Evaluating standard reconstruction (self-consistency check)...")
        std_metrics, std_aligned = evaluate_results(
            obj_true=std_obj_recon,  # Self comparison
            obj_recon=std_obj_recon,
            support=support,
            intensity=intensity,
            errors=std_errors,
            results_dir=std_results_dir
        )
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nAgent Metrics:")
        for k, v in agent_metrics.items():
            print(f"  {k}: {v:.6f}")
        
        print(f"\nStandard Metrics (self-consistency):")
        for k, v in std_metrics.items():
            print(f"  {k}: {v:.6f}")
        
        # Key metrics for comparison
        # cc_complex: Higher is better (measures correlation with reference)
        # r_factor: Lower is better (crystallographic R-factor)
        # phase_rmse_rad: Lower is better
        
        agent_cc = agent_metrics['cc_complex']
        agent_rfactor = agent_metrics['r_factor']
        agent_phase_rmse = agent_metrics['phase_rmse_rad']
        
        print("\n" + "="*60)
        print("PERFORMANCE VERIFICATION")
        print("="*60)
        
        # For phase retrieval, we check if the agent achieves reasonable reconstruction
        # Since it's an iterative algorithm with random initialization, we allow some variation
        
        # Check 1: Complex correlation should be high (> 0.7 for similar reconstruction)
        cc_threshold = 0.7
        cc_pass = agent_cc >= cc_threshold
        print(f"\nCC_complex: {agent_cc:.4f} (threshold >= {cc_threshold}) -> {'PASS' if cc_pass else 'FAIL'}")
        
        # Check 2: R-factor should be reasonable (< 0.5 typically)
        rfactor_threshold = 0.5
        rfactor_pass = agent_rfactor <= rfactor_threshold
        print(f"R-factor: {agent_rfactor:.4f} (threshold <= {rfactor_threshold}) -> {'PASS' if rfactor_pass else 'FAIL'}")
        
        # Check 3: Phase RMSE should be reasonable (< 1.0 rad typically)
        phase_rmse_threshold = 1.0
        phase_pass = agent_phase_rmse <= phase_rmse_threshold
        print(f"Phase RMSE: {agent_phase_rmse:.4f} rad (threshold <= {phase_rmse_threshold}) -> {'PASS' if phase_pass else 'FAIL'}")
        
        # Also check convergence - final error should be reasonable
        agent_final_error = agent_errors[-1] if agent_errors else float('inf')
        std_final_error = std_errors[-1] if std_errors else float('inf')
        
        print(f"\nFinal R-factor² (agent): {agent_final_error:.6f}")
        print(f"Final R-factor² (standard): {std_final_error:.6f}")
        
        # Allow agent's final error to be within 2x of standard
        error_ratio = agent_final_error / std_final_error if std_final_error > 0 else float('inf')
        error_pass = error_ratio <= 2.0
        print(f"Error ratio (agent/std): {error_ratio:.4f} (threshold <= 2.0) -> {'PASS' if error_pass else 'FAIL'}")
        
        # Overall pass/fail
        all_pass = cc_pass and rfactor_pass and phase_pass and error_pass
        
        print("\n" + "="*60)
        if all_pass:
            print("OVERALL: PASS - Agent performance is acceptable")
            print("="*60)
            sys.exit(0)
        else:
            print("OVERALL: FAIL - Agent performance degraded")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()