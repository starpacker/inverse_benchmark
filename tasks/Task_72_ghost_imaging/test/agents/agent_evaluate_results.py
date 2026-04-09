import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(data, reconstruction_results, results_dir):
    """
    Compute metrics and visualize reconstruction results.
    
    Args:
        data: dict from load_and_preprocess_data
        reconstruction_results: dict or list of dicts from run_inversion
        results_dir: directory to save results
    
    Returns:
        dict containing metrics for the best reconstruction
    """
    os.makedirs(results_dir, exist_ok=True)
    
    img_gt = data['img_gt']
    config = data['config']
    compression_ratio = config['compression_ratio']
    
    def compute_metrics(gt, rec):
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
    
    # Handle single result or list of results
    if isinstance(reconstruction_results, dict):
        reconstruction_results = [reconstruction_results]
    
    all_metrics = []
    for res in reconstruction_results:
        m = compute_metrics(img_gt, res['img_rec'])
        m['method'] = res['method']
        all_metrics.append(m)
        print(f"  {res['method']}: CC={m['CC']:.4f}, PSNR={m['PSNR']:.1f}")
    
    # Select best by CC
    best_idx = np.argmax([m['CC'] for m in all_metrics])
    best_result = reconstruction_results[best_idx]
    best_metrics = {k: v for k, v in all_metrics[best_idx].items() if k != 'method'}
    best_method = all_metrics[best_idx]['method']
    
    print(f"\n  → Using {best_method} (highest CC)")
    print("\n[Evaluation Metrics]:")
    for k, v in sorted(best_metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    
    # Find correlation and fista results for display
    img_corr = None
    img_fista = None
    for res in reconstruction_results:
        if res['method'] == 'correlation':
            img_corr = res['img_rec']
        elif res['method'] == 'fista':
            img_fista = res['img_rec']
    
    if img_corr is not None:
        rec_corr_n = img_corr / max(img_corr.max(), 1e-12)
        axes[1].imshow(rec_corr_n, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Correlation GI')
    else:
        axes[1].set_title('Correlation GI (N/A)')
    
    if img_fista is not None:
        rec_fista_n = img_fista / max(img_fista.max(), 1e-12)
        axes[2].imshow(rec_fista_n, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('FISTA CS')
        err = np.abs(img_gt - rec_fista_n)
    else:
        rec_best = best_result['img_rec']
        rec_best_n = rec_best / max(rec_best.max(), 1e-12)
        axes[2].imshow(rec_best_n, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'{best_method}')
        err = np.abs(img_gt - rec_best_n)
    
    axes[3].imshow(err, cmap='hot', vmin=0)
    axes[3].set_title('|Error|')
    
    for ax in axes:
        ax.axis('off')
    
    fig.suptitle(
        f"Ghost Imaging — Compressive Single-Pixel Reconstruction\n"
        f"M/N={compression_ratio:.0%} | PSNR={best_metrics['PSNR']:.1f} dB | "
        f"SSIM={best_metrics['SSIM']:.4f} | CC={best_metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_result['img_rec'])
    np.save(os.path.join(results_dir, "ground_truth.npy"), img_gt)
    
    return best_metrics
