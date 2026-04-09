import matplotlib

matplotlib.use('Agg')

import sys

import os

import json

import warnings

import numpy as np

import matplotlib.pyplot as plt

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')

sys.path.insert(0, REPO_DIR)

warnings.filterwarnings('ignore', message='Samples will be rescaled')

def evaluate_results(data, results, results_dir):
    """
    Evaluate and save the reconstruction results.
    
    Creates visualizations, saves arrays, and generates metrics JSON.
    
    Parameters
    ----------
    data : dict
        Output from load_and_preprocess_data
    results : dict
        Output from run_inversion
    results_dir : str
        Directory to save results
        
    Returns
    -------
    metrics : dict
        Dictionary of all evaluation metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    phantom = data['phantom']
    params = data['params']
    
    recon_adjoint = results['recon_adjoint']
    final_recon = results['recon_final']
    best_method = results['best_method']
    best_tv_label = results['best_tv_label']
    
    psnr_adj, ssim_adj, rmse_adj = results['metrics_adjoint']
    psnr_cg, ssim_cg, rmse_cg = results['metrics_cg']
    final_psnr, final_ssim, final_rmse = results['metrics_final']
    
    # Helper: normalize to [0, 1]
    def normalize_to_01(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    # Visualization
    gt_n = normalize_to_01(phantom)
    adj_n = normalize_to_01(recon_adjoint)
    iter_n = normalize_to_01(final_recon)
    error = np.abs(gt_n - iter_n)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    im0 = axes[0].imshow(gt_n, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(adj_n, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('DC Adjoint (Gridding)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(iter_n, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('CG + TV Recon', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Error Map', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Metric subtitles
    axes[1].text(0.5, -0.08, f'PSNR={psnr_adj:.1f}dB, SSIM={ssim_adj:.3f}',
                 transform=axes[1].transAxes, ha='center', fontsize=10)
    axes[2].text(0.5, -0.08, f'PSNR={final_psnr:.1f}dB, SSIM={final_ssim:.3f}',
                 transform=axes[2].transAxes, ha='center', fontsize=10)
    axes[3].text(0.5, -0.08, f'RMSE={final_rmse:.4f}',
                 transform=axes[3].transAxes, ha='center', fontsize=10)
    
    fig.suptitle('Non-Cartesian MRI Reconstruction via NUFFT (Radial Trajectory)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {vis_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), phantom)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), final_recon)
    print(f"  Saved ground_truth.npy and reconstruction.npy")
    
    # Compile metrics
    metrics = {
        'task': 'mri_nufft_recon',
        'task_number': 147,
        'inverse_problem': 'Non-Cartesian MRI reconstruction via NUFFT operators (radial trajectory)',
        'method': f'{best_tv_label} + TV denoising (CG on normal equations + Voronoi DC adjoint)',
        'library': 'mri-nufft + finufft',
        'image_size': params['N'],
        'n_spokes': params['n_spokes'],
        'nyquist_spokes': params['nyquist_spokes'],
        'acceleration_factor': round(params['acceleration'], 1),
        'noise_level': params['noise_level'],
        'adjoint_psnr': round(psnr_adj, 2),
        'adjoint_ssim': round(ssim_adj, 4),
        'adjoint_rmse': round(rmse_adj, 4),
        'cg_psnr': round(psnr_cg, 2),
        'cg_ssim': round(ssim_cg, 4),
        'cg_rmse': round(rmse_cg, 4),
        'psnr': round(final_psnr, 2),
        'ssim': round(final_ssim, 4),
        'rmse': round(final_rmse, 4),
        'best_method': best_method,
    }
    
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Best method: {best_method}")
    print(f"  PSNR:        {final_psnr:.2f} dB  (adjoint: {psnr_adj:.2f}, CG: {psnr_cg:.2f})")
    print(f"  SSIM:        {final_ssim:.4f}   (adjoint: {ssim_adj:.4f}, CG: {ssim_cg:.4f})")
    print(f"  RMSE:        {final_rmse:.4f}")
    print(f"  Status:      {'PASS' if final_psnr > 15 and final_ssim > 0.5 else 'FAIL'}")
    print("=" * 70)
    
    return metrics
