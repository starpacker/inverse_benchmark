import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============== INJECTED REFEREE CODE ==============
def compute_psnr(ref, test, data_range=None):
    """Compute Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = max(ref.max() - ref.min(), 1e-10)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    if mse < 1e-30:
        return 100.0
    return float(10 * np.log10(data_range**2 / mse))

def compute_rmse(ref, test):
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((ref - test)**2)))

def evaluate_results(all_dx_recon, all_dy_recon, dx_gt, dy_gt, 
                     grid_ys, grid_xs, ref_image, images, params):
    """
    Evaluate DIC reconstruction results and generate visualizations.
    
    Args:
        all_dx_recon: Recovered x-displacement fields (n_frames, ny, nx)
        all_dy_recon: Recovered y-displacement fields (n_frames, ny, nx)
        dx_gt: Ground truth x-displacement fields (n_frames, height, width)
        dy_gt: Ground truth y-displacement fields (n_frames, height, width)
        grid_ys: Y coordinates of grid points
        grid_xs: X coordinates of grid points
        ref_image: Reference speckle image
        images: Deformed image sequence
        params: Dictionary containing parameters
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    n_frames = len(all_dx_recon)
    
    # Extract ground truth at grid points
    all_dx_true = []
    all_dy_true = []
    for t in range(n_frames):
        dx_gt_s = dx_gt[t][np.ix_(grid_ys, grid_xs)]
        dy_gt_s = dy_gt[t][np.ix_(grid_ys, grid_xs)]
        all_dx_true.append(dx_gt_s)
        all_dy_true.append(dy_gt_s)
        
        errs = np.sqrt((all_dx_recon[t] - dx_gt_s)**2 + (all_dy_recon[t] - dy_gt_s)**2)
        print(f"       Frame {t:2d}: max_err={np.max(errs):.4f} px,  "
              f"mean_err={np.mean(errs):.4f} px")

    all_dx_true = np.array(all_dx_true)
    all_dy_true = np.array(all_dy_true)

    # Compute displacement magnitudes
    disp_true = np.sqrt(all_dx_true**2 + all_dy_true**2)
    disp_recon = np.sqrt(all_dx_recon**2 + all_dy_recon**2)

    # Compute metrics
    psnr_val = compute_psnr(disp_true, disp_recon)
    rmse_val = compute_rmse(disp_true, disp_recon)

    flat_t = disp_true.ravel()
    flat_r = disp_recon.ravel()
    if np.std(flat_t) > 1e-10 and np.std(flat_r) > 1e-10:
        cc_val = float(np.corrcoef(flat_t, flat_r)[0, 1])
    else:
        cc_val = 1.0 if np.allclose(flat_t, flat_r) else 0.0

    rmse_dx = compute_rmse(all_dx_true, all_dx_recon)
    rmse_dy = compute_rmse(all_dy_true, all_dy_recon)

    ssim_vals = []
    for t in range(n_frames):
        dr = max(disp_true[t].max() - disp_true[t].min(), 1e-10)
        s = ssim(disp_true[t], disp_recon[t], data_range=dr)
        ssim_vals.append(s)
    ssim_mean = float(np.mean(ssim_vals))

    metrics = {
        "psnr_dB": round(psnr_val, 2),
        "ssim": round(ssim_mean, 4),
        "correlation_coefficient": round(cc_val, 6),
        "rmse_displacement_pixels": round(rmse_val, 6),
        "rmse_dx_pixels": round(rmse_dx, 6),
        "rmse_dy_pixels": round(rmse_dy, 6),
        "n_frames": n_frames,
        "image_size": [params['height'], params['width']],
        "method": "ZNCC_integer_plus_LucasKanade_subpixel_DIC"
    }

    print(f"\n{'=' * 44}")
    print(f"  PSNR     = {psnr_val:.2f} dB")
    print(f"  SSIM     = {ssim_mean:.4f}")
    print(f"  CC       = {cc_val:.6f}")
    print(f"  RMSE     = {rmse_val:.6f} px")
    print(f"  RMSE(dx) = {rmse_dx:.6f} px")
    print(f"  RMSE(dy) = {rmse_dy:.6f} px")
    print(f"{'=' * 44}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] {metrics_path}")

    # ---- Visualization ----
    t_vis = int(np.argmax([np.max(np.abs(dx_gt[t])) for t in range(n_frames)]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    ax.imshow(ref_image, cmap='gray')
    ax.set_title('(a) Reference Speckle Image')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(images[t_vis], cmap='gray')
    ax.set_title(f'(b) Deformed Image (frame {t_vis})')
    ax.axis('off')

    ax = axes[0, 2]
    gt_mag = np.sqrt(dx_gt[t_vis]**2 + dy_gt[t_vis]**2)
    im = ax.imshow(gt_mag, cmap='hot')
    plt.colorbar(im, ax=ax, label='|d| (px)', shrink=0.8)
    ax.set_title(f'(c) GT Displacement Magnitude (frame {t_vis})')
    ax.axis('off')

    ax = axes[1, 0]
    ax.imshow(ref_image, cmap='gray', alpha=0.4)
    Y_grid, X_grid = np.meshgrid(grid_ys, grid_xs, indexing='ij')
    scale = 5
    ax.quiver(X_grid, Y_grid,
              all_dx_true[t_vis] * scale, all_dy_true[t_vis] * scale,
              color='blue', alpha=0.8, scale=1, scale_units='xy',
              label='Ground Truth')
    ax.quiver(X_grid + 1, Y_grid + 1,
              all_dx_recon[t_vis] * scale, all_dy_recon[t_vis] * scale,
              color='red', alpha=0.8, scale=1, scale_units='xy',
              label='DIC Recovered')
    ax.set_title(f'(d) Displacement Vectors (frame {t_vis})')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    ax = axes[1, 1]
    ax.scatter(disp_true.ravel(), disp_recon.ravel(),
               s=4, alpha=0.3, c='steelblue')
    lim = max(disp_true.max(), disp_recon.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', lw=1, label='Ideal')
    ax.set_xlabel('True |displacement| (px)')
    ax.set_ylabel('Recovered |displacement| (px)')
    ax.set_title(f'(e) True vs Recovered  (CC={cc_val:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    err_map = np.sqrt((all_dx_recon[t_vis] - all_dx_true[t_vis])**2 +
                      (all_dy_recon[t_vis] - all_dy_true[t_vis])**2)
    im = ax.imshow(err_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Error (px)', shrink=0.8)
    ax.set_title(f'(f) Displacement Error Map (frame {t_vis})')

    fig.suptitle(
        f"Task 140: pyidi_dic - DIC Displacement Tracking\n"
        f"PSNR={psnr_val:.2f} dB  |  SSIM={ssim_mean:.4f}  |  "
        f"CC={cc_val:.4f}  |  RMSE={rmse_val:.4f} px",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), disp_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), disp_recon)
    print(f"[SAVE] ground_truth.npy, recon_output.npy")

    return metrics
# ============== END INJECTED REFEREE CODE ==============


def main():
    # Data paths
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Load outer (primary) data
    if not outer_files:
        print("ERROR: No primary data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {outer_data.get('func_name', 'unknown')}")
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run agent's run_inversion
    print("\n[RUNNING] Agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("[SUCCESS] Agent's run_inversion completed.")
    except Exception as e:
        print(f"[FAILED] Agent's run_inversion raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner files (chained execution)
    if inner_files:
        print("\n[CHAINED EXECUTION] Detected inner data files.")
        # This would be for closure/factory pattern - not applicable here based on the function signature
        final_agent_result = agent_output
        final_std_result = std_output
    else:
        # Direct execution
        final_agent_result = agent_output
        final_std_result = std_output
    
    # Unpack results
    # run_inversion returns: (all_dx_recon, all_dy_recon, ys, xs)
    agent_dx, agent_dy, agent_ys, agent_xs = final_agent_result
    std_dx, std_dy, std_ys, std_xs = final_std_result
    
    print(f"\nAgent output shapes: dx={agent_dx.shape}, dy={agent_dy.shape}")
    print(f"Std output shapes: dx={std_dx.shape}, dy={std_dy.shape}")
    
    # For evaluation, we need ground truth data
    # The ground truth should be in the kwargs or we need to extract from args
    # Looking at the function signature: run_inversion(ref_image, images, subset_size=48, step=24, search_margin=6, lk_iterations=30)
    
    ref_image = args[0] if len(args) > 0 else kwargs.get('ref_image')
    images = args[1] if len(args) > 1 else kwargs.get('images')
    
    # We need dx_gt and dy_gt for evaluation. These might be passed separately or need to be inferred
    # Since this is a test, we compare agent vs std output directly
    
    # Check if ground truth displacement is available in kwargs
    dx_gt = kwargs.get('dx_gt', None)
    dy_gt = kwargs.get('dy_gt', None)
    
    if dx_gt is None or dy_gt is None:
        # Try to load from a separate file or infer from the test setup
        # For now, we'll use the standard output as a proxy for ground truth comparison
        print("\nWARNING: Ground truth displacement not found in kwargs.")
        print("Comparing agent output directly with standard output...")
        
        # Direct comparison approach
        rmse_dx = compute_rmse(std_dx, agent_dx)
        rmse_dy = compute_rmse(std_dy, agent_dy)
        
        disp_std = np.sqrt(std_dx**2 + std_dy**2)
        disp_agent = np.sqrt(agent_dx**2 + agent_dy**2)
        
        psnr_val = compute_psnr(disp_std, disp_agent)
        rmse_disp = compute_rmse(disp_std, disp_agent)
        
        flat_std = disp_std.ravel()
        flat_agent = disp_agent.ravel()
        if np.std(flat_std) > 1e-10 and np.std(flat_agent) > 1e-10:
            cc_val = float(np.corrcoef(flat_std, flat_agent)[0, 1])
        else:
            cc_val = 1.0 if np.allclose(flat_std, flat_agent) else 0.0
        
        print(f"\n{'=' * 50}")
        print("DIRECT COMPARISON (Agent vs Standard)")
        print(f"{'=' * 50}")
        print(f"  PSNR          = {psnr_val:.2f} dB")
        print(f"  CC            = {cc_val:.6f}")
        print(f"  RMSE (disp)   = {rmse_disp:.6f} px")
        print(f"  RMSE (dx)     = {rmse_dx:.6f} px")
        print(f"  RMSE (dy)     = {rmse_dy:.6f} px")
        print(f"{'=' * 50}")
        
        # Success criteria: agent should produce similar results to standard
        # Allow some tolerance
        if psnr_val >= 30.0 and cc_val >= 0.95:
            print("\n[PASS] Agent output is sufficiently close to standard output.")
            sys.exit(0)
        elif psnr_val >= 20.0 and cc_val >= 0.90:
            print("\n[PASS] Agent output is acceptable (within tolerance).")
            sys.exit(0)
        else:
            print("\n[FAIL] Agent output differs significantly from standard output.")
            sys.exit(1)
    else:
        # We have ground truth, use evaluate_results
        print("\nGround truth found. Running full evaluation...")
        
        # Build params dict
        params = {
            'height': ref_image.shape[0],
            'width': ref_image.shape[1]
        }
        
        print("\n[EVALUATING] Agent results...")
        agent_metrics = evaluate_results(
            agent_dx, agent_dy, dx_gt, dy_gt,
            agent_ys, agent_xs, ref_image, images, params
        )
        
        print("\n[EVALUATING] Standard results...")
        std_metrics = evaluate_results(
            std_dx, std_dy, dx_gt, dy_gt,
            std_ys, std_xs, ref_image, images, params
        )
        
        # Compare metrics (PSNR - higher is better)
        agent_psnr = agent_metrics['psnr_dB']
        std_psnr = std_metrics['psnr_dB']
        
        agent_ssim = agent_metrics['ssim']
        std_ssim = std_metrics['ssim']
        
        agent_cc = agent_metrics['correlation_coefficient']
        std_cc = std_metrics['correlation_coefficient']
        
        print(f"\n{'=' * 50}")
        print("FINAL COMPARISON")
        print(f"{'=' * 50}")
        print(f"  Agent PSNR: {agent_psnr:.2f} dB, Standard PSNR: {std_psnr:.2f} dB")
        print(f"  Agent SSIM: {agent_ssim:.4f}, Standard SSIM: {std_ssim:.4f}")
        print(f"  Agent CC: {agent_cc:.6f}, Standard CC: {std_cc:.6f}")
        print(f"{'=' * 50}")
        
        # Success criteria: agent should be within 10% of standard
        psnr_threshold = std_psnr * 0.9
        
        if agent_psnr >= psnr_threshold:
            print(f"\n[PASS] Agent PSNR ({agent_psnr:.2f}) >= threshold ({psnr_threshold:.2f})")
            sys.exit(0)
        else:
            print(f"\n[FAIL] Agent PSNR ({agent_psnr:.2f}) < threshold ({psnr_threshold:.2f})")
            sys.exit(1)


if __name__ == "__main__":
    main()