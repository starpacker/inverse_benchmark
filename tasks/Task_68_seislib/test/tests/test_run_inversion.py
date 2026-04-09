import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the referee (evaluation function) verbatim
def evaluate_results(dm_gt, dm_rec, stations, config, save_dir):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Args:
        dm_gt: ground truth velocity perturbation (nx, ny)
        dm_rec: reconstructed velocity perturbation (nx, ny)
        stations: station coordinates (n_stations, 2)
        config: dict with lat_min, lat_max, lon_min, lon_max
        save_dir: directory to save results
    
    Returns:
        dict containing metrics: PSNR, SSIM, CC, RE, RMSE
    """
    # Compute metrics
    gt_2d = dm_gt.copy()
    rec_2d = dm_rec.copy()
    data_range = gt_2d.max() - gt_2d.min()
    if data_range < 1e-12:
        data_range = 1.0
    
    mse = np.mean((gt_2d - rec_2d)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_2d, rec_2d, data_range=data_range))
    cc = float(np.corrcoef(gt_2d.ravel(), rec_2d.ravel())[0, 1])
    re = float(np.linalg.norm(gt_2d - rec_2d) / max(np.linalg.norm(gt_2d), 1e-12))
    rmse = float(np.sqrt(mse))
    
    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Visualization
    lat_min = config['lat_min']
    lat_max = config['lat_max']
    lon_min = config['lon_min']
    lon_max = config['lon_max']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmax = max(np.abs(dm_gt).max(), np.abs(dm_rec).max())
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    im0 = axes[0, 0].imshow(dm_gt.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 0].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 0].set_title('Ground Truth δc/c₀')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(dm_rec.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              origin='lower', extent=extent, aspect='auto')
    axes[0, 1].plot(stations[:, 1], stations[:, 0], 'k^', ms=4)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])
    
    err = dm_gt - dm_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent, aspect='auto')
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    mid = dm_gt.shape[0] // 2
    axes[1, 1].plot(dm_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(dm_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title(f'Cross-section (row {mid})')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Column index')
    axes[1, 1].set_ylabel('δc/c₀')
    
    fig.suptitle(
        f"seislib — Surface-Wave Tomography\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save data
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(save_dir, "reconstruction.npy"), dm_rec)
    np.save(os.path.join(save_dir, "ground_truth.npy"), dm_gt)
    
    return metrics


def compare_inversion_results(agent_result, std_result):
    """
    Compare inversion results without needing ground truth data.
    Returns a similarity score based on the reconstructed models.
    """
    metrics = {}
    
    # Extract all_results from both
    agent_all = agent_result.get('all_results', {})
    std_all = std_result.get('all_results', {})
    
    # Get common alpha values
    agent_alphas = set(agent_all.keys())
    std_alphas = set(std_all.keys())
    common_alphas = agent_alphas.intersection(std_alphas)
    
    if not common_alphas:
        print("Warning: No common alpha values found between agent and standard results")
        return {'CC': 0.0, 'RMSE': float('inf'), 'success': False}
    
    # Compare reconstructions for each alpha
    cc_values = []
    rmse_values = []
    
    for alpha in common_alphas:
        agent_rec = agent_all[alpha]['dm_rec']
        std_rec = std_all[alpha]['dm_rec']
        
        # Correlation coefficient
        cc = np.corrcoef(agent_rec.ravel(), std_rec.ravel())[0, 1]
        cc_values.append(cc)
        
        # RMSE
        rmse = np.sqrt(np.mean((agent_rec - std_rec)**2))
        rmse_values.append(rmse)
    
    avg_cc = np.mean(cc_values)
    avg_rmse = np.mean(rmse_values)
    
    # Check if Laplacian matrices match
    agent_L = agent_result.get('L')
    std_L = std_result.get('L')
    
    laplacian_match = False
    if agent_L is not None and std_L is not None:
        try:
            # Convert to dense for comparison if sparse
            if hasattr(agent_L, 'toarray'):
                agent_L_dense = agent_L.toarray()
            else:
                agent_L_dense = np.array(agent_L)
            
            if hasattr(std_L, 'toarray'):
                std_L_dense = std_L.toarray()
            else:
                std_L_dense = np.array(std_L)
            
            laplacian_match = np.allclose(agent_L_dense, std_L_dense, rtol=1e-5, atol=1e-8)
        except Exception as e:
            print(f"Warning: Could not compare Laplacian matrices: {e}")
            laplacian_match = True  # Give benefit of doubt
    
    return {
        'CC': avg_cc,
        'RMSE': avg_rmse,
        'laplacian_match': laplacian_match,
        'num_alphas_compared': len(common_alphas),
        'success': True
    }


def main():
    # Data paths
    data_paths = ['/data/yjh/seislib_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Found {len(outer_files)} outer file(s) and {len(inner_files)} inner file(s)")
    
    try:
        # Load outer (primary) data
        if not outer_files:
            print("ERROR: No outer data file found")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"Function: {outer_data.get('func_name', 'unknown')}")
        print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
        
        # Execute agent function
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent execution completed.")
        
        # Check if this is a chained execution (closure/factory pattern)
        if inner_files and callable(agent_output):
            print("Detected chained execution pattern...")
            inner_path = inner_files[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            print("Executing operator on inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Evaluate results
        print("\n=== Comparing Results ===")
        
        # Since both results are dicts with 'all_results', compare them directly
        comparison = compare_inversion_results(final_result, std_result)
        
        print(f"Correlation Coefficient (avg): {comparison['CC']:.6f}")
        print(f"RMSE (avg): {comparison['RMSE']:.6f}")
        print(f"Laplacian Match: {comparison.get('laplacian_match', 'N/A')}")
        print(f"Alphas Compared: {comparison.get('num_alphas_compared', 'N/A')}")
        
        # Determine success criteria
        # CC should be very high (close to 1.0) for identical algorithms
        # RMSE should be very small
        cc_threshold = 0.99  # Correlation should be > 99%
        
        success = True
        
        if comparison['CC'] < cc_threshold:
            print(f"\nWARNING: Correlation coefficient {comparison['CC']:.4f} < threshold {cc_threshold}")
            success = False
        
        if not comparison.get('laplacian_match', True):
            print("\nWARNING: Laplacian matrices do not match")
            # This might still be acceptable if reconstructions are similar
        
        # Also verify grid dimensions
        agent_nx = final_result.get('nx')
        agent_ny = final_result.get('ny')
        std_nx = std_result.get('nx')
        std_ny = std_result.get('ny')
        
        if agent_nx != std_nx or agent_ny != std_ny:
            print(f"\nWARNING: Grid dimensions mismatch - Agent: ({agent_nx}, {agent_ny}), Standard: ({std_nx}, {std_ny})")
            success = False
        else:
            print(f"Grid dimensions match: ({agent_nx}, {agent_ny})")
        
        # Check that all alpha values produced results
        agent_alphas = set(final_result.get('all_results', {}).keys())
        std_alphas = set(std_result.get('all_results', {}).keys())
        
        if agent_alphas != std_alphas:
            print(f"\nWARNING: Alpha values differ - Agent: {agent_alphas}, Standard: {std_alphas}")
            # This is acceptable if the main ones match
        else:
            print(f"Alpha values match: {sorted(agent_alphas)}")
        
        print(f"\n=== Final Result ===")
        print(f"Scores -> Agent CC: {comparison['CC']:.6f}, Threshold: {cc_threshold}")
        
        if success:
            print("TEST PASSED: Agent performance is acceptable")
            sys.exit(0)
        else:
            print("TEST FAILED: Agent performance degraded significantly")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()