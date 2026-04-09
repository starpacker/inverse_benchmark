import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(gt, recon):
    """Compute reconstruction quality metrics."""
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    # 3D PSNR
    mse = np.mean((gt_norm - recon_norm)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    # 3D correlation coefficient
    gt_flat = gt_norm.ravel()
    recon_flat = recon_norm.ravel()
    cc = np.corrcoef(gt_flat, recon_flat)[0, 1]

    # RMSE
    rmse = np.sqrt(mse)

    # Compute SSIM on central slices (2D SSIM)
    mid = gt.shape[0] // 2
    ssim_xy = structural_similarity(gt_norm[mid, :, :], recon_norm[mid, :, :],
                                     data_range=1.0)
    ssim_xz = structural_similarity(gt_norm[:, mid, :], recon_norm[:, mid, :],
                                     data_range=1.0)
    ssim_yz = structural_similarity(gt_norm[:, :, mid], recon_norm[:, :, mid],
                                     data_range=1.0)
    ssim_avg = (ssim_xy + ssim_xz + ssim_yz) / 3.0

    return {
        'psnr': float(psnr),
        'ssim_xy': float(ssim_xy),
        'ssim_xz': float(ssim_xz),
        'ssim_yz': float(ssim_yz),
        'ssim_avg': float(ssim_avg),
        'cc': float(cc),
        'rmse': float(rmse),
    }

def compute_fsc_aspire(gt_data, recon_data):
    """Compute Fourier Shell Correlation using ASPIRE."""
    from aspire.volume import Volume

    gt_vol = Volume(gt_data[np.newaxis, ...].astype(np.float64))
    recon_vol = Volume(recon_data[np.newaxis, ...].astype(np.float64))

    try:
        est_res, fsc_curve = gt_vol.fsc(recon_vol, cutoff=0.5)
        valid = fsc_curve[1:len(fsc_curve)//2]
        valid = valid[np.isfinite(valid)]
        mean_fsc = float(np.mean(valid)) if len(valid) > 0 else 0.0
        return mean_fsc, fsc_curve
    except Exception as e:
        print(f"  FSC computation failed: {e}")
        return None, None

def visualize_results_internal(gt, recon, metrics, method_name, save_path):
    """
    Create 6-panel visualization.
    """
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    mid = gt.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    slice_labels = ['Axial (XY)', 'Coronal (XZ)', 'Sagittal (YZ)']
    gt_slices = [gt_norm[mid, :, :], gt_norm[:, mid, :], gt_norm[:, :, mid]]
    recon_slices = [recon_norm[mid, :, :], recon_norm[:, mid, :], recon_norm[:, :, mid]]

    for j in range(3):
        im0 = axes[0, j].imshow(gt_slices[j], cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'GT {slice_labels[j]}', fontsize=12, fontweight='bold')
        axes[0, j].axis('off')
        plt.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)

        im1 = axes[1, j].imshow(recon_slices[j], cmap='gray', vmin=0, vmax=1)
        axes[1, j].set_title(f'Recon {slice_labels[j]}', fontsize=12, fontweight='bold')
        axes[1, j].axis('off')
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)

    metrics_text = (
        f"Method: {method_name}\n"
        f"PSNR: {metrics['psnr']:.2f} dB | CC: {metrics['cc']:.4f}\n"
        f"SSIM (avg): {metrics['ssim_avg']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )
    fig.suptitle(
        f"Cryo-EM 3D Reconstruction: {method_name}\n{metrics_text}",
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {save_path}")

def visualize_projections_internal(clean_images, noisy_images, save_path):
    """Show sample projection images (clean vs noisy)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    n_images = clean_images.shape[0]
    step = max(1, n_images // 4)

    for j in range(4):
        idx = min(j * step, n_images - 1)
        axes[0, j].imshow(clean_images[idx], cmap='gray')
        axes[0, j].set_title(f'Clean #{idx}', fontsize=11)
        axes[0, j].axis('off')

        axes[1, j].imshow(noisy_images[idx], cmap='gray')
        axes[1, j].set_title(f'Noisy #{idx}', fontsize=11)
        axes[1, j].axis('off')

    fig.suptitle('Sample Cryo-EM Projections (Clean vs Noisy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_results(data_dict, recon_me_result, recon_bp_result, n_projections, noise_var):
    """
    Evaluate reconstruction results and generate visualizations.
    
    Computes quality metrics (PSNR, SSIM, correlation, FSC), creates
    visualizations, and saves all results to disk.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        recon_me_result: Result dict from run_inversion (mean_estimator)
        recon_bp_result: Result dict from run_inversion (back_projection)
        n_projections: Number of projections used
        noise_var: Noise variance used
    
    Returns:
        Dictionary containing all metrics and evaluation results
    """
    gt_volume = data_dict['gt_volume']
    noisy_images = data_dict['noisy_images']
    clean_images = data_dict['clean_images']
    snr_db = data_dict['snr_db']
    vol_size = data_dict['vol_size']

    print("\n[5/5] Evaluation and visualization...")

    # Save sample projections
    visualize_projections_internal(
        clean_images, noisy_images,
        os.path.join(RESULTS_DIR, 'sample_projections.png')
    )

    me_success = recon_me_result['success']
    bp_success = recon_bp_result['success']

    recon_me = recon_me_result['recon_volume'] if me_success else None
    recon_bp = recon_bp_result['recon_volume'] if bp_success else None

    metrics_me = None
    metrics_bp = None

    if me_success:
        metrics_me = compute_metrics(gt_volume, recon_me)
        print(f"  MeanEstimator — PSNR: {metrics_me['psnr']:.2f} dB, "
              f"CC: {metrics_me['cc']:.4f}, SSIM: {metrics_me['ssim_avg']:.4f}")

    if bp_success:
        metrics_bp = compute_metrics(gt_volume, recon_bp)
        print(f"  BackProjection — PSNR: {metrics_bp['psnr']:.2f} dB, "
              f"CC: {metrics_bp['cc']:.4f}, SSIM: {metrics_bp['ssim_avg']:.4f}")

    # Determine primary reconstruction: pick the one with higher CC
    if me_success and bp_success:
        if metrics_me['cc'] > metrics_bp['cc']:
            primary_recon = recon_me
            primary_metrics = metrics_me
            primary_method = "ASPIRE MeanEstimator"
        else:
            primary_recon = recon_bp
            primary_metrics = metrics_bp
            primary_method = "Weighted Back-Projection"
    elif me_success:
        primary_recon = recon_me
        primary_metrics = metrics_me
        primary_method = "ASPIRE MeanEstimator"
    elif bp_success:
        primary_recon = recon_bp
        primary_metrics = metrics_bp
        primary_method = "Weighted Back-Projection"
    else:
        print("ERROR: All reconstruction methods failed!")
        return None

    # FSC with ASPIRE
    fsc_val, fsc_curve = compute_fsc_aspire(gt_volume, primary_recon)
    if fsc_val is not None:
        primary_metrics['fsc_mean'] = fsc_val
        print(f"  Mean FSC (primary): {fsc_val:.4f}")

    # Main visualization
    visualize_results_internal(
        gt_volume, primary_recon, primary_metrics,
        primary_method,
        os.path.join(RESULTS_DIR, 'vis_result.png')
    )

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, 'gt_volume.npy'), gt_volume)
    np.save(os.path.join(RESULTS_DIR, 'recon_volume.npy'), primary_recon)
    np.save(os.path.join(RESULTS_DIR, 'noisy_projections.npy'), noisy_images[:20])
    if me_success:
        np.save(os.path.join(RESULTS_DIR, 'recon_mean_estimator.npy'), recon_me)
    if bp_success:
        np.save(os.path.join(RESULTS_DIR, 'recon_back_projection.npy'), recon_bp)

    # Save metrics
    all_metrics = {
        'primary_method': primary_method,
        'volume_size': vol_size,
        'n_projections': n_projections,
        'noise_var': noise_var,
        'projection_snr_db': float(snr_db),
        'primary_metrics': primary_metrics,
    }
    if me_success:
        all_metrics['mean_estimator'] = metrics_me
    if bp_success:
        all_metrics['back_projection'] = metrics_bp

    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Primary method: {primary_method}")
    print(f"  Volume size: {vol_size}^3")
    print(f"  Projections: {n_projections}")
    print(f"  Projection SNR: {snr_db:.2f} dB")
    print(f"  PSNR: {primary_metrics['psnr']:.2f} dB")
    print(f"  SSIM (avg): {primary_metrics['ssim_avg']:.4f}")
    print(f"  Correlation: {primary_metrics['cc']:.4f}")
    print(f"  RMSE: {primary_metrics['rmse']:.4f}")
    if fsc_val is not None:
        print(f"  Mean FSC: {fsc_val:.4f}")

    if me_success and bp_success:
        print(f"\n  Comparison:")
        print(f"    MeanEstimator  — PSNR: {metrics_me['psnr']:.2f}, CC: {metrics_me['cc']:.4f}, SSIM: {metrics_me['ssim_avg']:.4f}")
        print(f"    BackProjection — PSNR: {metrics_bp['psnr']:.2f}, CC: {metrics_bp['cc']:.4f}, SSIM: {metrics_bp['ssim_avg']:.4f}")

    print("=" * 70)

    return all_metrics
