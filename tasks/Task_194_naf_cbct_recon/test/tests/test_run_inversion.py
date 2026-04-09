import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Inject the referee function (evaluate_results)
def evaluate_results(gt, recon, recon_full, sino_full, sino_sparse, 
                     angles_full, angles_sparse, out_dir):
    """
    Compute metrics (PSNR, SSIM, RMSE) and save visualization.
    
    Args:
        gt: ground truth 3D volume
        recon: sparse-view reconstructed 3D volume
        recon_full: full-view reconstructed 3D volume
        sino_full: full sinograms
        sino_sparse: sparse sinograms
        angles_full: full projection angles
        angles_sparse: sparse projection angles
        out_dir: output directory for saving results
        
    Returns:
        metrics_sparse: dict with PSNR, SSIM, RMSE for sparse reconstruction
        metrics_full: dict with PSNR, SSIM, RMSE for full reconstruction
    """
    def compute_metrics(gt_vol, recon_vol):
        """Compute PSNR, SSIM, RMSE on [0,1]-normalized data."""
        gt_f = gt_vol.astype(np.float64)
        recon_f = recon_vol.astype(np.float64)

        vmin, vmax = gt_f.min(), gt_f.max()
        if vmax - vmin > 1e-10:
            gt_n = (gt_f - vmin) / (vmax - vmin)
            recon_n = np.clip((recon_f - vmin) / (vmax - vmin), 0, 1)
        else:
            gt_n, recon_n = gt_f, recon_f

        rmse_val = np.sqrt(np.mean((gt_n - recon_n) ** 2))
        psnr_list, ssim_list = [], []
        for iz in range(gt_n.shape[0]):
            gs, rs = gt_n[iz], recon_n[iz]
            if gs.max() - gs.min() < 1e-8:
                continue
            psnr_list.append(psnr(gs, rs, data_range=1.0))
            ssim_list.append(ssim(gs, rs, data_range=1.0))

        return {
            'psnr': round(np.mean(psnr_list) if psnr_list else 0.0, 4),
            'ssim': round(np.mean(ssim_list) if ssim_list else 0.0, 4),
            'rmse': round(float(rmse_val), 6)
        }

    # Compute metrics
    metrics_full = compute_metrics(gt, recon_full)
    metrics_sparse = compute_metrics(gt, recon)

    # Visualization
    D, H, W = gt.shape
    md, mh, mw = D // 2, H // 2, W // 2

    vmin, vmax = gt.min(), gt.max()
    norm = lambda x: np.clip((x - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else x

    gd, rfd, rsd = norm(gt), norm(recon_full), norm(recon)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        'Sparse-View Cone-Beam CT Reconstruction\n'
        f'Sparse ({len(angles_sparse)} views): PSNR={metrics_sparse["psnr"]:.2f} dB, SSIM={metrics_sparse["ssim"]:.4f}  |  '
        f'Full ({len(angles_full)} views): PSNR={metrics_full["psnr"]:.2f} dB, SSIM={metrics_full["ssim"]:.4f}',
        fontsize=14, fontweight='bold')

    views = [('Axial', lambda x: x[md], 'equal'),
             ('Coronal', lambda x: x[:, mh, :], 'auto'),
             ('Sagittal', lambda x: x[:, :, mw], 'auto')]

    for row, (name, sl, asp) in enumerate(views):
        axes[row, 0].imshow(sl(gd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 0].set_title(f'GT ({name})'); axes[row, 0].axis('off')

        axes[row, 1].imshow(sl(rfd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 1].set_title(f'Full Recon ({name})'); axes[row, 1].axis('off')

        axes[row, 2].imshow(sl(rsd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 2].set_title(f'Sparse Recon ({name})'); axes[row, 2].axis('off')

        err = np.abs(sl(gd) - sl(rsd))
        im = axes[row, 3].imshow(err, cmap='hot', vmin=0, vmax=0.3, aspect=asp)
        axes[row, 3].set_title(f'Error ({name})'); axes[row, 3].axis('off')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    # Sinograms
    axes[3, 0].imshow(sino_full[md], cmap='gray', aspect='auto')
    axes[3, 0].set_title(f'Full Sinogram ({len(angles_full)} ang)'); axes[3, 0].set_xlabel('Angle')

    axes[3, 1].imshow(sino_sparse[md], cmap='gray', aspect='auto')
    axes[3, 1].set_title(f'Sparse Sinogram ({len(angles_sparse)} ang)'); axes[3, 1].set_xlabel('Angle')

    axes[3, 2].plot(gd[md, mh], 'k-', lw=2, label='GT')
    axes[3, 2].plot(rfd[md, mh], 'b--', lw=1.5, label=f'Full ({len(angles_full)})')
    axes[3, 2].plot(rsd[md, mh], 'r-.', lw=1.5, label=f'Sparse ({len(angles_sparse)})')
    axes[3, 2].legend(fontsize=8); axes[3, 2].set_title('Line Profile'); axes[3, 2].grid(alpha=0.3)

    axes[3, 3].axis('off')
    txt = (f"Volume: {D}x{H}x{W}\n\n"
           f"Full ({len(angles_full)} ang):\n  PSNR={metrics_full['psnr']:.2f}dB\n  SSIM={metrics_full['ssim']:.4f}\n  RMSE={metrics_full['rmse']:.6f}\n\n"
           f"Sparse ({len(angles_sparse)} ang):\n  PSNR={metrics_sparse['psnr']:.2f}dB\n  SSIM={metrics_sparse['ssim']:.4f}\n  RMSE={metrics_sparse['rmse']:.6f}\n\n"
           f"Method: FBP + SART")
    axes[3, 3].text(0.1, 0.95, txt, transform=axes[3, 3].transAxes, fontsize=11, va='top',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    vis_path = os.path.join(out_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {vis_path}")

    # Save metrics JSON
    out = {
        'task': 'naf_cbct_recon', 'task_id': 194, 'method': 'FBP_SART',
        'volume_size': D, 'num_angles_full': len(angles_full), 'num_angles_sparse': len(angles_sparse),
        'psnr': metrics_sparse['psnr'], 'ssim': metrics_sparse['ssim'], 'rmse': metrics_sparse['rmse'],
        'full_view_psnr': metrics_full['psnr'], 'full_view_ssim': metrics_full['ssim'], 'full_view_rmse': metrics_full['rmse'],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Save arrays
    np.save(os.path.join(out_dir, 'ground_truth.npy'), gt.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction.npy'), recon.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction_full.npy'), recon_full.astype(np.float32))

    return metrics_sparse, metrics_full


def compute_simple_metrics(gt_vol, recon_vol):
    """Compute PSNR, SSIM, RMSE on [0,1]-normalized data."""
    gt_f = gt_vol.astype(np.float64)
    recon_f = recon_vol.astype(np.float64)

    vmin, vmax = gt_f.min(), gt_f.max()
    if vmax - vmin > 1e-10:
        gt_n = (gt_f - vmin) / (vmax - vmin)
        recon_n = np.clip((recon_f - vmin) / (vmax - vmin), 0, 1)
    else:
        gt_n, recon_n = gt_f, recon_f

    rmse_val = np.sqrt(np.mean((gt_n - recon_n) ** 2))
    psnr_list, ssim_list = [], []
    for iz in range(gt_n.shape[0]):
        gs, rs = gt_n[iz], recon_n[iz]
        if gs.max() - gs.min() < 1e-8:
            continue
        psnr_list.append(psnr(gs, rs, data_range=1.0))
        ssim_list.append(ssim(gs, rs, data_range=1.0))

    return {
        'psnr': round(np.mean(psnr_list) if psnr_list else 0.0, 4),
        'ssim': round(np.mean(ssim_list) if ssim_list else 0.0, 4),
        'rmse': round(float(rmse_val), 6)
    }


def main():
    # Data paths provided
    data_paths = ['/data/yjh/naf_cbct_recon_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer (main) and inner (chained) data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No primary data file found.")
        sys.exit(1)
    
    print(f"Loading primary data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load primary data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs and expected output
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output')
    
    print(f"Function: {outer_data.get('func_name', 'run_inversion')}")
    print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's function
    print("\n--- Running Agent's run_inversion ---")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
    print(f"Standard output shape: {std_output.shape if hasattr(std_output, 'shape') else type(std_output)}")
    
    # Check if there are inner (chained) data files
    if inner_data_paths:
        print(f"\n--- Chained Execution Detected ---")
        print(f"Inner data files: {inner_data_paths}")
        
        # For chained execution, agent_output should be callable
        if callable(agent_output):
            for inner_path in inner_data_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_std_output = inner_data.get('output')
                
                # Execute the operator
                agent_final = agent_output(*inner_args, **inner_kwargs)
                std_final = inner_std_output
        else:
            # Direct execution
            agent_final = agent_output
            std_final = std_output
    else:
        # Direct execution (no chained calls)
        agent_final = agent_output
        std_final = std_output
    
    # Now we need to evaluate the results
    # The evaluate_results function expects: gt, recon, recon_full, sino_full, sino_sparse, angles_full, angles_sparse, out_dir
    # However, we only have the reconstruction output from run_inversion
    # We need to compare the agent's reconstruction quality against the standard's
    
    print("\n--- Evaluating Results ---")
    
    # Create output directory
    out_dir = './test_output'
    os.makedirs(out_dir, exist_ok=True)
    
    # Since run_inversion returns a reconstruction volume, we need to compare them
    # We'll use compute_simple_metrics to compare agent vs standard output
    
    # First, let's verify the outputs are comparable
    if not (hasattr(agent_final, 'shape') and hasattr(std_final, 'shape')):
        print("ERROR: Outputs are not arrays")
        sys.exit(1)
    
    if agent_final.shape != std_final.shape:
        print(f"WARNING: Shape mismatch - Agent: {agent_final.shape}, Standard: {std_final.shape}")
    
    # Compute metrics using std_final as ground truth reference
    # This checks if the agent's output is similar to the standard output
    metrics = compute_simple_metrics(std_final, agent_final)
    
    print(f"\n--- Comparison Metrics (Agent vs Standard) ---")
    print(f"  PSNR: {metrics['psnr']:.4f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    
    # Also compute self-metrics for the standard (should be perfect)
    std_self_metrics = compute_simple_metrics(std_final, std_final)
    print(f"\n--- Standard Self-Metrics (Reference) ---")
    print(f"  PSNR: {std_self_metrics['psnr']:.4f} dB (should be inf or very high)")
    print(f"  SSIM: {std_self_metrics['ssim']:.4f} (should be 1.0)")
    print(f"  RMSE: {std_self_metrics['rmse']:.6f} (should be 0.0)")
    
    # Determine success criteria
    # For reconstruction comparison, we expect high similarity
    # PSNR > 30 dB is generally considered good quality
    # SSIM > 0.9 is considered good structural similarity
    
    psnr_threshold = 25.0  # Minimum PSNR in dB
    ssim_threshold = 0.85  # Minimum SSIM
    
    print(f"\n--- Quality Assessment ---")
    print(f"  PSNR threshold: {psnr_threshold} dB")
    print(f"  SSIM threshold: {ssim_threshold}")
    
    # Check if metrics meet thresholds
    psnr_ok = metrics['psnr'] >= psnr_threshold or np.isinf(metrics['psnr'])
    ssim_ok = metrics['ssim'] >= ssim_threshold
    
    if psnr_ok and ssim_ok:
        print(f"\n✓ PASSED: Agent output meets quality thresholds")
        print(f"  PSNR: {metrics['psnr']:.4f} >= {psnr_threshold}")
        print(f"  SSIM: {metrics['ssim']:.4f} >= {ssim_threshold}")
        sys.exit(0)
    else:
        print(f"\n✗ FAILED: Agent output does not meet quality thresholds")
        if not psnr_ok:
            print(f"  PSNR: {metrics['psnr']:.4f} < {psnr_threshold}")
        if not ssim_ok:
            print(f"  SSIM: {metrics['ssim']:.4f} < {ssim_threshold}")
        sys.exit(1)


if __name__ == '__main__':
    main()