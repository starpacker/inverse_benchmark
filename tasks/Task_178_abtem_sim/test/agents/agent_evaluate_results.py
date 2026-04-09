import json

import os

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def normalize(x):
    """Normalize array to [0, 1] range."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-15)

def evaluate_results(data_dict, result_dict):
    """
    Evaluate the inversion results and save outputs.
    
    Computes quality metrics (PSNR, SSIM, correlation coefficient, relative errors)
    and saves results to disk including metrics JSON, numpy arrays, and visualization.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing preprocessed data including ground truth
    result_dict : dict
        Dictionary containing inversion results
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics
    """
    print("[5/5] Evaluating ...")
    
    gt_img = data_dict['gt_img']
    noisy_img = data_dict['noisy_img']
    V_pot = data_dict['V_pot']
    true_df = data_dict['true_defocus']
    true_t = data_dict['true_thickness']
    
    recon_img = result_dict['recon_img']
    est_df = result_dict['estimated_defocus']
    est_t = result_dict['estimated_thickness']
    
    # Normalize images for metric computation
    gt_n = normalize(gt_img)
    recon_n = normalize(recon_img)
    noisy_n = normalize(noisy_img)

    # Compute quality metrics for reconstruction
    psnr_r = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
    ssim_r = structural_similarity(gt_n, recon_n, data_range=1.0)
    cc_r = float(np.corrcoef(gt_n.ravel(), recon_n.ravel())[0, 1])

    # Compute quality metrics for noisy observation
    psnr_n = peak_signal_noise_ratio(gt_n, noisy_n, data_range=1.0)
    ssim_n = structural_similarity(gt_n, noisy_n, data_range=1.0)

    # Compute relative errors for parameter estimation
    re_df = abs(est_df - true_df) / abs(true_df)
    re_t = abs(est_t - true_t) / abs(true_t)

    # Print results
    print(f"\n{'─'*55}")
    print(f"  TRUE : df={true_df:.1f} nm, t={true_t:.2f} nm")
    print(f"  EST  : df={est_df:.3f} nm, t={est_t:.4f} nm")
    print(f"  RE   : df={re_df:.6f}, t={re_t:.6f}")
    print(f"  Recon: PSNR={psnr_r:.2f}, SSIM={ssim_r:.4f}, CC={cc_r:.4f}")
    print(f"  Noisy: PSNR={psnr_n:.2f}, SSIM={ssim_n:.4f}")
    print(f"{'─'*55}\n")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Prepare metrics dictionary
    metrics = {
        "task_id": 178,
        "task_name": "abtem_sim",
        "inverse_problem": "HRTEM inverse parameter estimation (defocus + thickness)",
        "true_defocus_nm": true_df,
        "true_thickness_nm": true_t,
        "estimated_defocus_nm": round(float(est_df), 3),
        "estimated_thickness_nm": round(float(est_t), 4),
        "defocus_relative_error": round(float(re_df), 6),
        "thickness_relative_error": round(float(re_t), 6),
        "reconstruction_PSNR_dB": round(float(psnr_r), 2),
        "reconstruction_SSIM": round(float(ssim_r), 4),
        "reconstruction_CC": round(float(cc_r), 4),
        "noisy_PSNR_dB": round(float(psnr_n), 2),
        "noisy_SSIM": round(float(ssim_n), 4),
    }
    
    # Save metrics JSON
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/metrics.json")

    # Save numpy arrays
    np.save("results/ground_truth.npy", gt_img)
    np.save("results/noisy_observation.npy", noisy_img)
    np.save("results/reconstruction.npy", recon_img)
    np.save("results/projected_potential.npy", V_pot)
    print("Saved .npy arrays")

    # Create visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    im0 = axes[0, 0].imshow(gt_n, cmap="gray", origin="lower")
    axes[0, 0].set_title("(a) GT noiseless HRTEM image", fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(noisy_n, cmap="gray", origin="lower")
    axes[0, 1].set_title("(b) Noisy observation", fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(recon_n, cmap="gray", origin="lower")
    axes[1, 0].set_title(f"(c) Best-fit image\ndf={est_df:.2f} nm, t={est_t:.3f} nm", fontsize=11)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    err = np.abs(gt_n - recon_n)
    im3 = axes[1, 1].imshow(err, cmap="hot", origin="lower")
    axes[1, 1].set_title(f"(d) |GT − Recon| error\nPSNR={psnr_r:.1f} dB, SSIM={ssim_r:.3f}", fontsize=11)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Task 178: abtem_sim — HRTEM Inverse Parameter Estimation\n"
        f"True: df={true_df} nm, t={true_t} nm  |  Est: df={est_df:.2f} nm, t={est_t:.3f} nm\n"
        f"Defocus RE={re_df:.4f}, Thickness RE={re_t:.4f}",
        fontsize=12, fontweight="bold", y=1.01)

    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig("results/reconstruction_result.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/reconstruction_result.png\n\nDone.")

    return metrics
