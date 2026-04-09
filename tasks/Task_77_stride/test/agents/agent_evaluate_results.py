import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(c_gt, inversion_results, transducers, domain_size, results_dir):
    """
    Evaluate reconstruction quality and select the best result.
    Compute metrics, generate visualizations, and save results.
    
    Args:
        c_gt: ground truth sound speed (nx, ny)
        inversion_results: dict from run_inversion
        transducers: transducer positions
        domain_size: physical domain size
        results_dir: directory to save results
        
    Returns:
        dict containing:
            - metrics: quality metrics (PSNR, SSIM, CC, RE, RMSE)
            - best_alpha: best regularisation parameter
            - c_rec: best reconstruction
    """
    all_results = inversion_results['all_results']
    alpha_list = inversion_results['alpha_list']

    best_cc = -1
    best_rec = None
    best_alpha = alpha_list[0]

    print("\n[INVERSION RESULTS]")
    for alpha in alpha_list:
        c_rec = all_results[alpha]
        cc_val = float(np.corrcoef(c_gt.ravel(), c_rec.ravel())[0, 1])
        print(f"  α={alpha:7.2f} → CC={cc_val:.4f}")
        if cc_val > best_cc:
            best_cc = cc_val
            best_alpha = alpha
            best_rec = c_rec

    print(f"  → Best α={best_alpha} with CC={best_cc:.4f}")

    # Compute metrics
    data_range = c_gt.max() - c_gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    mse = np.mean((c_gt - best_rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(c_gt, best_rec, data_range=data_range))
    cc = float(np.corrcoef(c_gt.ravel(), best_rec.ravel())[0, 1])
    re = float(np.linalg.norm(c_gt - best_rec) / max(np.linalg.norm(c_gt), 1e-12))
    rmse = float(np.sqrt(mse))

    metrics = {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

    print("\n[EVALUATION METRICS]")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), c_gt)

    # Visualization
    nx, ny = c_gt.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent = [-domain_size / 2 * 1000, domain_size / 2 * 1000,
              -domain_size / 2 * 1000, domain_size / 2 * 1000]

    vmin = min(c_gt.min(), best_rec.min())
    vmax = max(c_gt.max(), best_rec.max())

    n_transducers = len(transducers)

    # GT
    im0 = axes[0, 0].imshow(c_gt.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 0].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 0].set_title('Ground Truth c(x,y) [m/s]')
    axes[0, 0].set_xlabel('x [mm]')
    axes[0, 0].set_ylabel('y [mm]')
    plt.colorbar(im0, ax=axes[0, 0])

    # Reconstruction
    im1 = axes[0, 1].imshow(best_rec.T, cmap='jet', vmin=vmin, vmax=vmax,
                              origin='lower', extent=extent)
    axes[0, 1].plot(transducers[:, 0] * 1000, transducers[:, 1] * 1000,
                     'k.', ms=3)
    axes[0, 1].set_title('LSQR Reconstruction')
    plt.colorbar(im1, ax=axes[0, 1])

    # Error
    err = c_gt - best_rec
    im2 = axes[1, 0].imshow(err.T, cmap='RdBu_r', origin='lower',
                              extent=extent)
    axes[1, 0].set_title('Error (GT - Recon)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Profile
    mid = c_gt.shape[0] // 2
    x_mm = np.linspace(-domain_size / 2, domain_size / 2, c_gt.shape[0]) * 1000
    axes[1, 1].plot(x_mm, c_gt[mid, :], 'b-', lw=2, label='GT')
    axes[1, 1].plot(x_mm, best_rec[mid, :], 'r--', lw=2, label='Recon')
    axes[1, 1].set_title('Central Profile')
    axes[1, 1].set_xlabel('x [mm]')
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(
        f"stride — USCT Sound-Speed Tomography ({n_transducers} transducers)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'metrics': metrics,
        'best_alpha': best_alpha,
        'c_rec': best_rec
    }
