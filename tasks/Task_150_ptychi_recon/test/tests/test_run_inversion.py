import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import the target function
from agent_run_inversion import run_inversion

# Define results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# INJECT THE REFEREE (evaluate_results) - Copied from Reference B
# ============================================================================

def plot_results(gt, recon, metrics, errors, path):
    """Generate visualization of reconstruction results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im0 = axes[0, 0].imshow(np.abs(gt), cmap='gray')
    axes[0, 0].set_title('GT Amplitude', fontsize=14)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.angle(gt), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('GT Phase', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].semilogy(errors, 'b-', lw=1.5)
    axes[0, 2].set(xlabel='Iteration', ylabel='Error',
                    title='ePIE Convergence')
    axes[0, 2].grid(True, alpha=0.3)

    im2 = axes[1, 0].imshow(np.abs(recon), cmap='gray')
    axes[1, 0].set_title(
        f'Recon Amplitude\nPSNR={metrics["psnr"]:.2f} dB  '
        f'SSIM={metrics["ssim"]:.4f}', fontsize=14)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(np.angle(recon), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(
        f'Recon Phase\nPhase corr={metrics["phase_correlation"]:.4f}',
        fontsize=14)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    amp_err = np.abs(np.abs(gt) - np.abs(recon))
    im4 = axes[1, 2].imshow(amp_err, cmap='hot')
    axes[1, 2].set_title(f'Amplitude Error\nRMSE={metrics["rmse"]:.4f}',
                          fontsize=14)
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        if ax is not axes[0, 2]:
            ax.axis('off')

    plt.suptitle('Ptychographic Reconstruction (ePIE)\n'
                 f'Complex corr: {metrics["complex_correlation"]:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot -> {path}")

def evaluate_results(gt_object, recon_result, n_iter, n_positions, overlap, photon_count):
    """
    Evaluate reconstruction quality and save results.

    Resolves the global complex-factor ambiguity of ptychography and computes
    PSNR / SSIM / RMSE on the amplitude image, plus phase-error metrics.

    Parameters
    ----------
    gt_object : ndarray (complex)
        Ground truth complex object.
    recon_result : dict
        Dictionary from run_inversion containing 'object', 'probe', 'errors'.
    n_iter : int
        Number of iterations used.
    n_positions : int
        Number of scan positions.
    overlap : float
        Overlap fraction used.
    photon_count : float
        Photon count used.

    Returns
    -------
    dict
        Dictionary containing all metrics and aligned reconstruction.
    """
    recon = recon_result['object']
    errors = recon_result['errors']
    gt = gt_object

    # Global complex scaling: a = <gt, recon> / <recon, recon>
    a = np.sum(gt * np.conj(recon)) / (np.sum(np.abs(recon)**2) + 1e-30)
    recon_a = recon * a

    gt_amp = np.abs(gt)
    rc_amp = np.abs(recon_a)

    # Per-pixel amplitude alignment via linear fit
    mask = gt_amp > 0.05 * gt_amp.max()
    if mask.sum() > 10:
        c = np.polyfit(rc_amp[mask].ravel(), gt_amp[mask].ravel(), 1)
        rc_amp_s = np.clip(c[0] * rc_amp + c[1], 0, None)
    else:
        rc_amp_s = rc_amp

    # Normalise to [0, 1] using GT range
    lo, hi = gt_amp.min(), gt_amp.max()
    gt_n = (gt_amp - lo) / (hi - lo + 1e-10)
    rc_n = np.clip((rc_amp_s - lo) / (hi - lo + 1e-10), 0, 1)

    # Compute amplitude metrics
    p = float(psnr(gt_n, rc_n, data_range=1.0))
    s = float(ssim(gt_n, rc_n, data_range=1.0))
    r = float(np.sqrt(np.mean((gt_n - rc_n)**2)))

    # Phase metrics
    gt_ph = np.angle(gt)
    rc_ph = np.angle(recon_a)
    if mask.sum() > 0:
        diff = np.angle(np.exp(1j * (rc_ph[mask] - gt_ph[mask])))
        offset = np.median(diff)
        rc_ph_c = rc_ph - offset
        diff2 = np.angle(np.exp(1j * (rc_ph_c[mask] - gt_ph[mask])))
        ph_err = float(np.sqrt(np.mean(diff2**2)))
        ph_corr = float(np.corrcoef(gt_ph[mask].ravel(),
                                     rc_ph_c[mask].ravel())[0, 1])
    else:
        ph_err, ph_corr = float('inf'), 0.0
        rc_ph_c = rc_ph

    # Complex correlation
    cc = float(np.abs(np.sum(recon_a * np.conj(gt))) /
               (np.sqrt(np.sum(np.abs(recon_a)**2) *
                        np.sum(np.abs(gt)**2)) + 1e-30))

    # Aligned reconstruction
    recon_aligned = rc_amp_s * np.exp(1j * rc_ph_c)

    metrics = {
        'psnr': p,
        'ssim': s,
        'rmse': r,
        'phase_error_rad': ph_err,
        'phase_correlation': ph_corr,
        'complex_correlation': cc
    }

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), np.abs(gt))
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), np.abs(recon_aligned))
    np.save(os.path.join(RESULTS_DIR, "gt_complex.npy"), gt)
    np.save(os.path.join(RESULTS_DIR, "recon_complex.npy"), recon_aligned)

    metrics_out = {
        **metrics,
        'n_iterations': n_iter,
        'n_scan_positions': n_positions,
        'overlap': overlap,
        'photon_count': photon_count
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Generate visualization
    plot_results(gt, recon_aligned, metrics, errors,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    return {
        'metrics': metrics,
        'recon_aligned': recon_aligned,
        'errors': errors
    }


# ============================================================================
# MAIN TEST LOGIC
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/ptychi_recon_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No primary data file found.")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"\nLoading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args length: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Fix random seed for reproducibility
        np.random.seed(42)
        
        # Run the agent's implementation
        print("\n" + "="*60)
        print("Running AGENT implementation...")
        print("="*60)
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Chained execution pattern
            inner_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # agent_output should be callable
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Extract evaluation parameters from the inputs
        # args order based on function signature:
        # patterns, positions, probe_init, obj_shape, n_iter, alpha_obj, alpha_probe, probe_update_start
        n_iter = kwargs.get('n_iter', 150)
        if 'n_iter' not in kwargs and len(args) > 4:
            n_iter = args[4]
        
        positions = args[1] if len(args) > 1 else kwargs.get('positions', [])
        n_positions = len(positions)
        
        # Default values for overlap and photon_count (not directly in function signature)
        overlap = 0.6  # Typical default
        photon_count = 1e6  # Typical default
        
        # We need a ground truth object for evaluation
        # The standard output should contain the reconstructed object
        # For evaluation, we'll use the standard result's object as a reference
        # to compare quality metrics
        
        print("\n" + "="*60)
        print("Evaluating AGENT results...")
        print("="*60)
        
        # Check if we have the expected output structure
        if std_result is None:
            print("WARNING: No standard result available for comparison.")
            print("Checking agent output structure...")
            
            if isinstance(final_result, dict) and 'object' in final_result:
                print("Agent output has correct structure.")
                print(f"  - Object shape: {final_result['object'].shape}")
                print(f"  - Probe shape: {final_result['probe'].shape}")
                print(f"  - Errors length: {len(final_result['errors'])}")
                print(f"  - Final error: {final_result['errors'][-1]:.4e}")
                sys.exit(0)
            else:
                print("ERROR: Agent output has unexpected structure.")
                sys.exit(1)
        
        # Both results should be dictionaries with 'object', 'probe', 'errors'
        if not isinstance(final_result, dict) or 'object' not in final_result:
            print(f"ERROR: Agent result has unexpected format: {type(final_result)}")
            sys.exit(1)
        
        if not isinstance(std_result, dict) or 'object' not in std_result:
            print(f"ERROR: Standard result has unexpected format: {type(std_result)}")
            sys.exit(1)
        
        # Use standard result's object as ground truth for comparison
        # This is a relative comparison - we're checking if agent performs similarly
        gt_object = std_result['object']
        
        # Evaluate agent's reconstruction against standard
        print("\n" + "="*60)
        print("Evaluating AGENT reconstruction quality...")
        print("="*60)
        
        agent_eval = evaluate_results(
            gt_object=gt_object,
            recon_result=final_result,
            n_iter=n_iter,
            n_positions=n_positions,
            overlap=overlap,
            photon_count=photon_count
        )
        
        # For standard, evaluate against itself (should be perfect)
        print("\n" + "="*60)
        print("Evaluating STANDARD reconstruction quality (self-comparison)...")
        print("="*60)
        
        std_eval = evaluate_results(
            gt_object=gt_object,
            recon_result=std_result,
            n_iter=n_iter,
            n_positions=n_positions,
            overlap=overlap,
            photon_count=photon_count
        )
        
        # Extract primary metrics for comparison
        agent_psnr = agent_eval['metrics']['psnr']
        agent_ssim = agent_eval['metrics']['ssim']
        agent_cc = agent_eval['metrics']['complex_correlation']
        
        std_psnr = std_eval['metrics']['psnr']
        std_ssim = std_eval['metrics']['ssim']
        std_cc = std_eval['metrics']['complex_correlation']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"  PSNR  -> Agent: {agent_psnr:.2f} dB, Standard: {std_psnr:.2f} dB")
        print(f"  SSIM  -> Agent: {agent_ssim:.4f}, Standard: {std_ssim:.4f}")
        print(f"  CC    -> Agent: {agent_cc:.4f}, Standard: {std_cc:.4f}")
        
        # Also compare convergence (final error)
        agent_final_err = final_result['errors'][-1]
        std_final_err = std_result['errors'][-1]
        print(f"  Final Error -> Agent: {agent_final_err:.4e}, Standard: {std_final_err:.4e}")
        
        # Verification logic
        # Since we're comparing agent to standard as ground truth,
        # agent should achieve high similarity (PSNR > threshold, SSIM > threshold)
        
        # For PSNR/SSIM, higher is better
        # We expect very high values since we're comparing reconstructions
        # The standard compared to itself should be perfect (infinite PSNR, SSIM=1)
        
        # For the agent, we check if it achieves reasonable reconstruction quality
        # compared to the standard output
        
        min_psnr_threshold = 20.0  # dB - reasonable reconstruction quality
        min_ssim_threshold = 0.8   # Good structural similarity
        min_cc_threshold = 0.9     # High complex correlation
        
        # Allow 10% margin for numerical differences
        margin = 0.90
        
        success = True
        failure_reasons = []
        
        # Check PSNR - should be reasonably high
        if agent_psnr < min_psnr_threshold:
            failure_reasons.append(f"PSNR ({agent_psnr:.2f}) below threshold ({min_psnr_threshold})")
            success = False
        
        # Check SSIM
        if agent_ssim < min_ssim_threshold:
            failure_reasons.append(f"SSIM ({agent_ssim:.4f}) below threshold ({min_ssim_threshold})")
            success = False
        
        # Check complex correlation
        if agent_cc < min_cc_threshold:
            failure_reasons.append(f"Complex correlation ({agent_cc:.4f}) below threshold ({min_cc_threshold})")
            success = False
        
        # Check if agent's convergence is reasonable (within 2x of standard)
        if agent_final_err > std_final_err * 2.0:
            failure_reasons.append(f"Final error ({agent_final_err:.4e}) significantly worse than standard ({std_final_err:.4e})")
            # This is a warning, not a failure
            print(f"WARNING: {failure_reasons[-1]}")
            failure_reasons.pop()
        
        print("\n" + "="*60)
        if success:
            print("TEST PASSED: Agent implementation meets quality thresholds.")
            print("="*60)
            sys.exit(0)
        else:
            print("TEST FAILED: Agent implementation did not meet quality thresholds.")
            for reason in failure_reasons:
                print(f"  - {reason}")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()