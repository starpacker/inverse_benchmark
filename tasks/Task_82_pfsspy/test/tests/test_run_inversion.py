import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from scipy.ndimage import zoom

# Define the results directory
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ===== INJECTED REFEREE CODE =====
def evaluate_results(br_clean, br_noisy, output, br_recon, rss, results_dir):
    """
    Evaluate PFSS reconstruction quality and generate visualizations.
    
    Compare the reconstructed photospheric B_r (from PFSS solution)
    with the clean (noise-free) input magnetogram.
    Also evaluate field properties at different heights.
    
    Args:
        br_clean: Clean (noise-free) ground truth magnetogram
        br_noisy: Noisy input magnetogram
        output: PFSS output object
        br_recon: Reconstructed B_r at photosphere
        rss: Source surface radius
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Get magnetic field components from PFSS output
    bg_raw = output.bg  # (nphi+1, ns+1, nr+1, 3) - B field on cell boundaries
    bg = np.array(bg_raw.value if hasattr(bg_raw, 'value') else bg_raw)
    
    # Resize clean to match recon shape if needed
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_resized = zoom(br_clean, zoom_factors)
    else:
        br_clean_resized = br_clean
    
    # Flatten for comparison
    gt = br_clean_resized.flatten()
    recon = br_recon.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((gt - recon)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt, recon)[0, 1]
    
    # Relative error
    re = np.sqrt(np.mean((gt - recon)**2)) / np.sqrt(np.mean(gt**2))
    
    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - recon)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # Magnetic energy (proxy for field quality)
    # Total unsigned flux at photosphere
    total_flux_gt = np.sum(np.abs(gt))
    total_flux_recon = np.sum(np.abs(recon))
    flux_ratio = total_flux_recon / total_flux_gt if total_flux_gt > 0 else 0
    
    # B_r at source surface (should be ~0 for open field)
    br_ss_raw = output.bc[0][:, :, -1]
    br_ss = np.array(br_ss_raw.value if hasattr(br_ss_raw, 'value') else br_ss_raw).T
    max_br_ss = np.max(np.abs(br_ss))
    
    # Open flux
    open_flux = np.sum(np.abs(br_ss))
    
    metrics = {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'relative_error': float(re),
        'flux_ratio': float(flux_ratio),
        'max_br_source_surface': float(max_br_ss),
        'open_flux': float(open_flux),
        'br_recon_shape': list(br_recon.shape),
        'bg_shape': list(bg.shape),
    }
    
    # Print metrics
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.4f} G")
    print(f"[EVAL] Relative Error = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Flux ratio = {metrics['flux_ratio']:.4f}")
    print(f"[EVAL] Max B_r at source surface = {metrics['max_br_source_surface']:.4f} G")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), br_clean)
    np.save(os.path.join(results_dir, "reconstruction.npy"), br_recon)
    np.save(os.path.join(results_dir, "input.npy"), br_noisy)
    print(f"[SAVE] GT shape: {br_clean.shape} → ground_truth.npy")
    print(f"[SAVE] Recon shape: {br_recon.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {br_noisy.shape} → input.npy")
    
    # Generate visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    _visualize_results(br_clean, br_noisy, br_recon, br_ss, metrics, rss, vis_path)
    
    return metrics

def _visualize_results(br_clean, br_noisy, br_recon, br_ss, metrics, rss, save_path):
    """Generate comprehensive PFSS visualization (helper function)."""
    # Resize clean and noisy for comparison
    if br_clean.shape != br_recon.shape:
        zoom_factors = [br_recon.shape[i] / br_clean.shape[i] for i in range(2)]
        br_clean_r = zoom(br_clean, zoom_factors)
        br_noisy_r = zoom(br_noisy, zoom_factors)
    else:
        br_clean_r = br_clean
        br_noisy_r = br_noisy
    
    vmax = max(np.abs(br_clean_r).max(), np.abs(br_recon).max())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Clean magnetogram
    im0 = axes[0, 0].imshow(br_clean_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 0].set_title('GT Magnetogram (clean)')
    axes[0, 0].set_xlabel('Longitude (px)')
    axes[0, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im0, ax=axes[0, 0], label='B_r (G)')
    
    # (b) Noisy magnetogram (input)
    im1 = axes[0, 1].imshow(br_noisy_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 1].set_title('Input Magnetogram (noisy)')
    axes[0, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im1, ax=axes[0, 1], label='B_r (G)')
    
    # (c) Reconstructed B_r at photosphere
    im2 = axes[0, 2].imshow(br_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto', origin='lower')
    axes[0, 2].set_title('PFSS Reconstructed B_r')
    axes[0, 2].set_xlabel('Longitude (px)')
    plt.colorbar(im2, ax=axes[0, 2], label='B_r (G)')
    
    # (d) Error map
    error = br_clean_r - br_recon
    im3 = axes[1, 0].imshow(error, cmap='seismic', 
                             vmin=-vmax*0.3, vmax=vmax*0.3,
                             aspect='auto', origin='lower')
    axes[1, 0].set_title('Error (GT - Recon)')
    axes[1, 0].set_xlabel('Longitude (px)')
    axes[1, 0].set_ylabel('Sine Latitude (px)')
    plt.colorbar(im3, ax=axes[1, 0], label='ΔB_r (G)')
    
    # (e) B_r at source surface
    im4 = axes[1, 1].imshow(br_ss, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1, 1].set_title(f'B_r at Source Surface (R={rss} R_sun)')
    axes[1, 1].set_xlabel('Longitude (px)')
    plt.colorbar(im4, ax=axes[1, 1], label='B_r (G)')
    
    # (f) Scatter: GT vs Recon
    axes[1, 2].scatter(br_clean_r.flatten(), br_recon.flatten(), 
                       alpha=0.3, s=5, c='steelblue')
    lim = vmax * 1.1
    axes[1, 2].plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Identity')
    axes[1, 2].set_xlabel('GT B_r (G)')
    axes[1, 2].set_ylabel('Recon B_r (G)')
    axes[1, 2].set_title(f'GT vs Recon (CC={metrics["cc"]:.4f})')
    axes[1, 2].legend()
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xlim([-lim, lim])
    axes[1, 2].set_ylim([-lim, lim])
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pfsspy — PFSS Coronal Magnetic Field Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} G | RE={metrics['relative_error']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")

# ===== END REFEREE CODE =====


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pfsspy_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Parse file paths to identify execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    try:
        # Load the primary (outer) data
        print(f"\n[LOAD] Loading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Number of args: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent's implementation
        print("\n[RUN] Executing agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("[RUN] Agent execution completed.")
        
        # Determine if we have chained execution
        if inner_data_paths:
            # Chained execution - agent_output should be callable
            print("\n[INFO] Chained execution detected.")
            inner_data_path = inner_data_paths[0]
            print(f"[LOAD] Loading inner data from: {inner_data_path}")
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("[RUN] Executing chained function...")
            final_result = agent_output(*inner_args, **inner_kwargs)
            print("[RUN] Chained execution completed.")
        else:
            # Direct execution
            print("\n[INFO] Direct execution pattern.")
            final_result = agent_output
            std_result = std_output
        
        # Extract results - run_inversion returns (output, br_recon)
        print("\n[INFO] Processing results...")
        
        # Agent results
        if isinstance(final_result, tuple) and len(final_result) == 2:
            agent_pfss_output, agent_br_recon = final_result
        else:
            print(f"[ERROR] Unexpected agent result format: {type(final_result)}")
            sys.exit(1)
        
        # Standard results
        if isinstance(std_result, tuple) and len(std_result) == 2:
            std_pfss_output, std_br_recon = std_result
        else:
            print(f"[ERROR] Unexpected standard result format: {type(std_result)}")
            sys.exit(1)
        
        # Extract input parameters for evaluation
        # args[0] is br_map (sunpy Map), args[1] is nr, args[2] is rss
        br_map = args[0]
        nr = args[1] if len(args) > 1 else kwargs.get('nr', 25)
        rss = args[2] if len(args) > 2 else kwargs.get('rss', 2.5)
        
        # Extract br_clean and br_noisy from the map
        # The br_map data is the noisy input
        br_noisy = np.array(br_map.data)
        
        # For evaluation, we use br_recon as both clean and noisy since
        # we're comparing reconstructions. The "clean" ground truth would
        # be the standard reconstruction.
        br_clean = std_br_recon  # Use standard as ground truth
        
        # Create separate results directories for agent and standard
        agent_results_dir = os.path.join(RESULTS_DIR, "agent")
        std_results_dir = os.path.join(RESULTS_DIR, "standard")
        os.makedirs(agent_results_dir, exist_ok=True)
        os.makedirs(std_results_dir, exist_ok=True)
        
        # Evaluate agent results
        print("\n" + "="*60)
        print("EVALUATING AGENT RESULTS")
        print("="*60)
        agent_metrics = evaluate_results(
            br_clean=br_clean,
            br_noisy=br_noisy,
            output=agent_pfss_output,
            br_recon=agent_br_recon,
            rss=rss,
            results_dir=agent_results_dir
        )
        
        # Evaluate standard results (for comparison)
        print("\n" + "="*60)
        print("EVALUATING STANDARD RESULTS")
        print("="*60)
        std_metrics = evaluate_results(
            br_clean=br_clean,
            br_noisy=br_noisy,
            output=std_pfss_output,
            br_recon=std_br_recon,
            rss=rss,
            results_dir=std_results_dir
        )
        
        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # Primary metrics for comparison
        agent_psnr = agent_metrics['psnr']
        std_psnr = std_metrics['psnr']
        agent_cc = agent_metrics['cc']
        std_cc = std_metrics['cc']
        agent_rmse = agent_metrics['rmse']
        std_rmse = std_metrics['rmse']
        
        print(f"Scores -> Agent PSNR: {agent_psnr:.4f}, Standard PSNR: {std_psnr:.4f}")
        print(f"Scores -> Agent CC: {agent_cc:.6f}, Standard CC: {std_cc:.6f}")
        print(f"Scores -> Agent RMSE: {agent_rmse:.4f}, Standard RMSE: {std_rmse:.4f}")
        
        # Determine success
        # PSNR and CC: Higher is better
        # RMSE: Lower is better
        # Allow 10% margin for PSNR and CC, and RMSE should not be more than 10% worse
        
        psnr_threshold = std_psnr * 0.9 if std_psnr > 0 else std_psnr * 1.1
        cc_threshold = std_cc * 0.9 if std_cc > 0 else std_cc * 1.1
        rmse_threshold = std_rmse * 1.1 if std_rmse > 0 else std_rmse * 0.9
        
        psnr_pass = agent_psnr >= psnr_threshold or np.isinf(agent_psnr)
        cc_pass = agent_cc >= cc_threshold
        rmse_pass = agent_rmse <= rmse_threshold
        
        print(f"\n[CHECK] PSNR: {agent_psnr:.4f} >= {psnr_threshold:.4f}? {'PASS' if psnr_pass else 'FAIL'}")
        print(f"[CHECK] CC: {agent_cc:.6f} >= {cc_threshold:.6f}? {'PASS' if cc_pass else 'FAIL'}")
        print(f"[CHECK] RMSE: {agent_rmse:.4f} <= {rmse_threshold:.4f}? {'PASS' if rmse_pass else 'FAIL'}")
        
        # Overall pass if at least 2 of 3 metrics pass, or if all critical metrics are close
        passes = sum([psnr_pass, cc_pass, rmse_pass])
        
        if passes >= 2:
            print("\n[RESULT] PASS - Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("\n[RESULT] FAIL - Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()