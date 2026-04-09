import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

# Inject the Referee (Evaluation Logic)
def evaluate_results(ground_truth, reconstruction, r_arr, z_arr, 
                     measurements, n_detectors, n_los_per_det, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics (PSNR, SSIM, RMSE), generates visualization,
    and saves all artifacts to disk.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground truth emissivity field (NR, NZ)
    reconstruction : ndarray
        Reconstructed emissivity field (NR, NZ)
    r_arr : ndarray
        1D array of R coordinates
    z_arr : ndarray
        1D array of Z coordinates
    measurements : ndarray
        Noisy measurements (for sinogram visualization)
    n_detectors : int
        Number of detector fans
    n_los_per_det : int
        LOS per detector
    results_dir : str
        Directory to save results
        
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, RMSE values
    """
    print("[5/6] Computing metrics …")
    
    gt_2d = ground_truth
    recon_2d = reconstruction
    
    # Normalise reconstruction to same scale as GT for fair comparison
    if recon_2d.max() > 0:
        recon_2d = recon_2d / recon_2d.max() * gt_2d.max()
    
    # Compute metrics
    data_range = gt_2d.max() - gt_2d.min()
    if data_range == 0:
        data_range = 1.0
    
    psnr = peak_signal_noise_ratio(gt_2d, recon_2d, data_range=data_range)
    ssim = structural_similarity(gt_2d, recon_2d, data_range=data_range)
    rmse = np.sqrt(np.mean((gt_2d - recon_2d) ** 2))
    
    metrics = {
        "PSNR": round(float(psnr), 4),
        "SSIM": round(float(ssim), 4),
        "RMSE": round(float(rmse), 6)
    }
    
    print(f"       PSNR = {metrics['PSNR']:.2f} dB")
    print(f"       SSIM = {metrics['SSIM']:.4f}")
    print(f"       RMSE = {metrics['RMSE']:.6f}")
    
    # Save results
    print("[6/6] Saving results …")
    error_map = recon_2d - gt_2d
    
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_2d)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_2d)
    np.save(os.path.join(results_dir, "measurements.npy"), measurements)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics → {os.path.join(results_dir, 'metrics.json')}")
    
    # Generate visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    _visualise(r_arr, z_arr, gt_2d, measurements, recon_2d, error_map, 
               metrics, n_detectors, n_los_per_det, vis_path)
    
    print("\n✓ Pipeline complete.")
    
    return metrics

def _visualise(r_arr, z_arr, gt, sinogram, recon, error, metrics, 
               n_detectors, n_los_per_det, save_path):
    """4-panel figure: GT, sinogram, reconstruction, error map."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    extent_rz = [r_arr[0], r_arr[-1], z_arr[0], z_arr[-1]]

    # (a) Ground truth
    ax = axes[0, 0]
    im = ax.imshow(gt.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title("(a) Ground Truth Emissivity ε(R,Z)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) Line-integrated measurements (sinogram-like)
    ax = axes[0, 1]
    n_det = n_detectors
    n_los = n_los_per_det
    sino_2d = sinogram.reshape(n_det, n_los) if len(sinogram) == n_det * n_los \
        else sinogram.reshape(-1, n_los)[:n_det]
    im = ax.imshow(sino_2d, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("(b) Line-Integrated Measurements")
    ax.set_xlabel("LOS index")
    ax.set_ylabel("Detector fan")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) Reconstruction
    ax = axes[1, 0]
    im = ax.imshow(recon.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="inferno")
    ax.set_title(f"(c) Reconstruction  PSNR={metrics['PSNR']:.1f} dB  "
                 f"SSIM={metrics['SSIM']:.3f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (d) Error map
    ax = axes[1, 1]
    im = ax.imshow(error.T, origin="lower", extent=extent_rz,
                   aspect="auto", cmap="seismic",
                   vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    ax.set_title(f"(d) Error Map  RMSE={metrics['RMSE']:.4f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Plasma/Fusion Tomography — Tokamak Emissivity Reconstruction",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualisation → {save_path}")


def compute_metrics_only(ground_truth, reconstruction):
    """
    Compute metrics without saving files - for comparison purposes.
    """
    gt_2d = ground_truth.copy()
    recon_2d = reconstruction.copy()
    
    # Normalise reconstruction to same scale as GT for fair comparison
    if recon_2d.max() > 0:
        recon_2d = recon_2d / recon_2d.max() * gt_2d.max()
    
    # Compute metrics
    data_range = gt_2d.max() - gt_2d.min()
    if data_range == 0:
        data_range = 1.0
    
    psnr = peak_signal_noise_ratio(gt_2d, recon_2d, data_range=data_range)
    ssim = structural_similarity(gt_2d, recon_2d, data_range=data_range)
    rmse = np.sqrt(np.mean((gt_2d - recon_2d) ** 2))
    
    return {
        "PSNR": round(float(psnr), 4),
        "SSIM": round(float(ssim), 4),
        "RMSE": round(float(rmse), 6)
    }


def main():
    # Data paths provided
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs")
        
        # Execute the agent's run_inversion
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner files (chained execution)
        if inner_files:
            # Chained execution pattern
            print("Detected chained execution pattern")
            inner_path = inner_files[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("Detected direct execution pattern")
            final_result = agent_output
            std_result = std_output
        
        print(f"Agent result shape: {final_result.shape if hasattr(final_result, 'shape') else type(final_result)}")
        print(f"Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # For run_inversion, we need ground truth to evaluate
        # The standard result IS the expected reconstruction
        # We compare agent reconstruction with standard reconstruction
        
        # Compute metrics comparing agent output to standard output
        # Treat standard output as "ground truth" for comparison
        metrics_agent = compute_metrics_only(std_result, final_result)
        
        # For the standard vs itself (should be perfect)
        metrics_std = compute_metrics_only(std_result, std_result)
        
        print(f"\nScores -> Agent: PSNR={metrics_agent['PSNR']}, SSIM={metrics_agent['SSIM']}, RMSE={metrics_agent['RMSE']}")
        print(f"Scores -> Standard: PSNR={metrics_std['PSNR']}, SSIM={metrics_std['SSIM']}, RMSE={metrics_std['RMSE']}")
        
        # For PSNR and SSIM: Higher is better
        # For RMSE: Lower is better
        
        # Check if agent performance is acceptable
        # Since we're comparing to the standard output, PSNR should be very high if outputs match
        # Allow some margin for numerical differences
        
        psnr_threshold = 30.0  # PSNR above 30 dB indicates very good match
        ssim_threshold = 0.95  # SSIM above 0.95 indicates very high similarity
        
        success = True
        
        if metrics_agent['PSNR'] < psnr_threshold:
            print(f"WARNING: Agent PSNR ({metrics_agent['PSNR']}) below threshold ({psnr_threshold})")
            # Check if it's still reasonable
            if metrics_agent['PSNR'] < 20.0:
                success = False
                print("FAIL: PSNR too low, significant quality degradation")
        
        if metrics_agent['SSIM'] < ssim_threshold:
            print(f"WARNING: Agent SSIM ({metrics_agent['SSIM']}) below threshold ({ssim_threshold})")
            if metrics_agent['SSIM'] < 0.8:
                success = False
                print("FAIL: SSIM too low, significant structural difference")
        
        # Also check direct numerical comparison
        if hasattr(final_result, 'shape') and hasattr(std_result, 'shape'):
            if final_result.shape == std_result.shape:
                max_diff = np.max(np.abs(final_result - std_result))
                mean_diff = np.mean(np.abs(final_result - std_result))
                print(f"Direct comparison - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                
                # If very close numerically, that's good
                if max_diff < 1e-6:
                    print("Results are numerically identical (within 1e-6)")
                    success = True
                elif max_diff < 0.01:
                    print("Results are very close (within 0.01)")
                    success = True
        
        if success:
            print("\n✓ TEST PASSED: Agent performance is acceptable")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent performance degraded significantly")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()