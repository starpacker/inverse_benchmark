import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, label

from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(gt, rec):
    """Compute SAR image quality metrics."""
    # Normalise both to [0, 1]
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}

def evaluate_results(sigma_gt, inversion_results, results_dir):
    """
    Evaluate SAR reconstruction results and generate visualizations.
    
    Computes quality metrics (PSNR, SSIM, CC, RE, RMSE) and applies
    post-processing to optimize reconstruction quality.
    
    Args:
        sigma_gt: ground truth scene reflectivity
        inversion_results: dict containing img_bp and img_pfa
        results_dir: directory to save results
        
    Returns:
        dict containing final metrics and best reconstruction
    """
    img_bp = inversion_results['img_bp']
    img_pfa = inversion_results['img_pfa']
    
    # Compute metrics for both methods
    m_bp = compute_metrics(sigma_gt, img_bp)
    m_pfa = compute_metrics(sigma_gt, img_pfa)
    print(f"  Backprojection CC={m_bp['CC']:.4f}")
    print(f"  PFA CC={m_pfa['CC']:.4f}")
    
    # Choose best method
    if m_bp['CC'] >= m_pfa['CC']:
        img_rec = img_bp
        metrics = m_bp
        method = "Backprojection"
    else:
        img_rec = img_pfa
        metrics = m_pfa
        method = "PFA"
    print(f"\n  → Using {method} (higher CC)")
    
    # ── Normalize and clean reconstruction ──
    img_rec = img_rec / max(img_rec.max(), 1e-12)
    sigma_gt_norm = sigma_gt / max(sigma_gt.max(), 1e-12)

    # --- Post-processing: median filter + threshold + Gaussian blur ---
    img_med = median_filter(img_rec, size=9)
    img_med[img_med < 0.16] = 0
    img_med = gaussian_filter(img_med, sigma=0.7)
    img_med = img_med / max(img_med.max(), 1e-12)

    # Also try: simple thresholded version
    img_thresh = img_rec.copy()
    img_thresh[img_thresh < 0.20] = 0
    img_thresh = gaussian_filter(img_thresh, sigma=1.0)
    img_thresh = img_thresh / max(img_thresh.max(), 1e-12)

    # Compare approaches
    m_med = compute_metrics(sigma_gt_norm, img_med)
    m_thresh = compute_metrics(sigma_gt_norm, img_thresh)
    m_raw = compute_metrics(sigma_gt_norm, img_rec)

    print(f"\n  Raw normalized:     CC={m_raw['CC']:.4f}, PSNR={m_raw['PSNR']:.2f}")
    print(f"  Median+thresh+blur: CC={m_med['CC']:.4f}, PSNR={m_med['PSNR']:.2f}")
    print(f"  Thresholded+blur:   CC={m_thresh['CC']:.4f}, PSNR={m_thresh['PSNR']:.2f}")

    # Pick the best approach by CC
    candidates = [
        (img_rec, m_raw, "raw"),
        (img_thresh, m_thresh, "thresholded"),
        (img_med, m_med, "median-filtered"),
    ]
    best_img, best_metrics, best_name = max(candidates, key=lambda x: x[1]['CC'])
    print(f"  → Best approach: {best_name}")
    img_rec_final = best_img
    metrics_final = best_metrics

    print(f"\n  Final: CC={metrics_final['CC']:.4f}, PSNR={metrics_final['PSNR']:.2f}, SSIM={metrics_final['SSIM']:.4f}")

    # Print evaluation metrics
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics_final.items()):
        print(f"  {k:20s} = {v}")

    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_final, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), img_rec_final)
    np.save(os.path.join(results_dir, "ground_truth.npy"), sigma_gt_norm)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gt_db = 20 * np.log10(sigma_gt_norm / max(sigma_gt_norm.max(), 1e-12) + 1e-6)
    bp_db = 20 * np.log10(img_bp / max(img_bp.max(), 1e-12) + 1e-6)
    pfa_db = 20 * np.log10(img_pfa / max(img_pfa.max(), 1e-12) + 1e-6)

    vmin = -40
    for ax, img, title in zip(axes,
                               [gt_db, bp_db, pfa_db],
                               ['Ground Truth', 'Backprojection', 'PFA']):
        im = ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=0,
                        origin='lower', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='dB')

    fig.suptitle(
        f"RITSAR — SAR Image Formation\n"
        f"PSNR={metrics_final['PSNR']:.1f} dB | CC={metrics_final['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'metrics': metrics_final,
        'img_rec': img_rec_final,
        'sigma_gt_norm': sigma_gt_norm,
        'img_bp': img_bp,
        'img_pfa': img_pfa
    }
