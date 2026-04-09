import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

# Import target function
from agent_run_inversion import run_inversion


def evaluate_results(gt, recon, sinogram, results_dir, assets_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes metrics:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - RMSE: Root Mean Square Error
    - CC: Pearson Correlation Coefficient
    
    Also generates visualization and saves all outputs.
    
    Parameters
    ----------
    gt : ndarray
        Ground truth attenuation map
    recon : ndarray
        Reconstructed attenuation map
    sinogram : ndarray
        Sinogram used for reconstruction
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
        
    Returns
    -------
    metrics : dict
        Dictionary with PSNR, SSIM, RMSE, CC values
    """
    # Crop to same size (iradon may produce slightly different shape)
    min_h = min(gt.shape[0], recon.shape[0])
    min_w = min(gt.shape[1], recon.shape[1])
    gt_c = gt[:min_h, :min_w]
    re_c = recon[:min_h, :min_w]

    # RMSE
    rmse = np.sqrt(np.mean((gt_c - re_c)**2))

    # Data range
    data_range = gt_c.max() - gt_c.min()
    if data_range < 1e-10:
        data_range = 1.0

    # PSNR
    mse = np.mean((gt_c - re_c)**2)
    psnr = 10 * np.log10(data_range**2 / (mse + 1e-12))

    # SSIM
    ssim_val = ssim(gt_c, re_c, data_range=data_range)

    # CC (Pearson correlation)
    g = gt_c.flatten() - gt_c.mean()
    r = re_c.flatten() - re_c.mean()
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "RMSE": float(rmse),
        "CC": float(cc),
    }

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    im0 = axes[0, 0].imshow(sinogram.T, cmap="gray", aspect="auto",
                             extent=[0, 180, -sinogram.shape[0]//2, sinogram.shape[0]//2])
    axes[0, 0].set_title("Sinogram (neutron transmission)", fontsize=14)
    axes[0, 0].set_xlabel("Angle (degrees)")
    axes[0, 0].set_ylabel("Detector position")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(gt_n, cmap="inferno")
    axes[0, 1].set_title("Ground Truth (μ distribution)", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(re_n, cmap="inferno")
    axes[1, 0].set_title(
        f"FBP Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"SSIM={metrics['SSIM']:.4f}",
        fontsize=12,
    )
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Crop for error visualization
    min_h_vis = min(gt_n.shape[0], re_n.shape[0])
    min_w_vis = min(gt_n.shape[1], re_n.shape[1])
    error = np.abs(gt_n[:min_h_vis, :min_w_vis] - re_n[:min_h_vis, :min_w_vis])
    im3 = axes[1, 1].imshow(error, cmap="magma")
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.4f} cm⁻¹)", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    
    # Save figures
    for p in [os.path.join(results_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "reconstruction_result.png"),
              os.path.join(assets_dir, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()

    # Save data
    for d in [results_dir, assets_dir]:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon)
        np.save(os.path.join(d, "ground_truth.npy"), gt)
        np.save(os.path.join(d, "reconstruction.npy"), recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main():
    data_paths = ['/data/yjh/neutompy_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    # Run the agent's function
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("Agent run_inversion executed successfully.")
    except Exception as e:
        print(f"ERROR running agent run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Check for chained execution
    if len(inner_paths) > 0:
        # Chained execution pattern
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output

    # Extract sinogram from args for evaluate_results
    sinogram = args[0] if len(args) > 0 else kwargs.get('sinogram', None)
    if sinogram is None:
        print("ERROR: Could not extract sinogram from input data.")
        sys.exit(1)

    # Use std_result as ground truth for evaluation
    # The evaluate_results function compares gt vs recon
    # gt = std_result (ground truth reconstruction), recon = agent reconstruction
    gt = std_result

    # Create output directories
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    assets_dir = os.path.join(os.path.dirname(__file__), "test_assets")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Evaluate agent output against standard output
    try:
        metrics_agent = evaluate_results(gt, final_result, sinogram, results_dir, assets_dir)
        print(f"\nAgent Metrics (vs Ground Truth):")
        print(f"  PSNR: {metrics_agent['PSNR']:.4f} dB")
        print(f"  SSIM: {metrics_agent['SSIM']:.6f}")
        print(f"  RMSE: {metrics_agent['RMSE']:.6f}")
        print(f"  CC:   {metrics_agent['CC']:.6f}")
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Also evaluate standard output against itself (perfect score baseline)
    results_dir_std = os.path.join(os.path.dirname(__file__), "test_results_std")
    assets_dir_std = os.path.join(os.path.dirname(__file__), "test_assets_std")
    os.makedirs(results_dir_std, exist_ok=True)
    os.makedirs(assets_dir_std, exist_ok=True)

    try:
        metrics_std = evaluate_results(gt, std_result, sinogram, results_dir_std, assets_dir_std)
        print(f"\nStandard Metrics (vs Ground Truth - should be perfect):")
        print(f"  PSNR: {metrics_std['PSNR']:.4f} dB")
        print(f"  SSIM: {metrics_std['SSIM']:.6f}")
        print(f"  RMSE: {metrics_std['RMSE']:.6f}")
        print(f"  CC:   {metrics_std['CC']:.6f}")
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Performance verification
    # PSNR and SSIM: Higher is better
    # RMSE: Lower is better
    # CC: Higher is better (closer to 1)

    print(f"\nScores -> Agent PSNR: {metrics_agent['PSNR']:.4f}, Standard PSNR: {metrics_std['PSNR']:.4f}")
    print(f"Scores -> Agent SSIM: {metrics_agent['SSIM']:.6f}, Standard SSIM: {metrics_std['SSIM']:.6f}")
    print(f"Scores -> Agent RMSE: {metrics_agent['RMSE']:.6f}, Standard RMSE: {metrics_std['RMSE']:.6f}")
    print(f"Scores -> Agent CC:   {metrics_agent['CC']:.6f}, Standard CC:   {metrics_std['CC']:.6f}")

    # Since we're comparing agent output to the standard output as ground truth,
    # perfect agent should have infinite PSNR, SSIM=1, RMSE=0, CC=1
    # We check that PSNR is very high (agent matches standard closely)
    
    passed = True

    # Check PSNR - should be very high if outputs match
    # A PSNR > 30 dB generally means very good match
    if metrics_agent['PSNR'] < 30.0:
        print(f"FAIL: Agent PSNR ({metrics_agent['PSNR']:.4f}) is too low (< 30 dB)")
        passed = False

    # Check SSIM - should be very close to 1
    if metrics_agent['SSIM'] < 0.90:
        print(f"FAIL: Agent SSIM ({metrics_agent['SSIM']:.6f}) is too low (< 0.90)")
        passed = False

    # Check CC - should be very close to 1
    if metrics_agent['CC'] < 0.90:
        print(f"FAIL: Agent CC ({metrics_agent['CC']:.6f}) is too low (< 0.90)")
        passed = False

    # Additional direct comparison: check numpy arrays are close
    try:
        if final_result.shape == std_result.shape:
            max_diff = np.max(np.abs(final_result - std_result))
            mean_diff = np.mean(np.abs(final_result - std_result))
            print(f"\nDirect comparison: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
            if max_diff < 1e-6:
                print("Arrays are essentially identical.")
        else:
            print(f"\nShape mismatch: agent={final_result.shape}, std={std_result.shape}")
            # Still might be acceptable if metrics are good
    except Exception as e:
        print(f"Direct comparison failed: {e}")

    if passed:
        print("\n=== TEST PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TEST FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()