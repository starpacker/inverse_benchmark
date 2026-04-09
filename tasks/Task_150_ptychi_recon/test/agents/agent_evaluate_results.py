import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_results(gt, recon, metrics, errors, path):
    """Generate visualization of reconstruction results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im0 = axes[0, 0].imshow(np.abs(gt), cmap='gray')
    axes[0, 0].set_title('GT Amplitude', fontsize=14)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.angle(gt), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('GT Phase', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].semilogy(errors, 'b-', lw=1.5)
    axes[0, 2].set(xlabel='Iteration', ylabel='Error',
                    title='ePIE Convergence')
    axes[0, 2].grid(True, alpha=0.3)

    im2 = axes[1, 0].imshow(np.abs(recon), cmap='gray')
    axes[1, 0].set_title(
        f'Recon Amplitude\nPSNR={metrics["psnr"]:.2f} dB  '
        f'SSIM={metrics["ssim"]:.4f}', fontsize=14)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(np.angle(recon), cmap='twilight',
                              vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(
        f'Recon Phase\nPhase corr={metrics["phase_correlation"]:.4f}',
        fontsize=14)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    amp_err = np.abs(np.abs(gt) - np.abs(recon))
    im4 = axes[1, 2].imshow(amp_err, cmap='hot')
    axes[1, 2].set_title(f'Amplitude Error\nRMSE={metrics["rmse"]:.4f}',
                          fontsize=14)
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        if ax is not axes[0, 2]:
            ax.axis('off')

    plt.suptitle('Ptychographic Reconstruction (ePIE)\n'
                 f'Complex corr: {metrics["complex_correlation"]:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot -> {path}")

def evaluate_results(gt_object, recon_result, n_iter, n_positions, overlap, photon_count):
    """
    Evaluate reconstruction quality and save results.

    Resolves the global complex-factor ambiguity of ptychography and computes
    PSNR / SSIM / RMSE on the amplitude image, plus phase-error metrics.

    Parameters
    ----------
    gt_object : ndarray (complex)
        Ground truth complex object.
    recon_result : dict
        Dictionary from run_inversion containing 'object', 'probe', 'errors'.
    n_iter : int
        Number of iterations used.
    n_positions : int
        Number of scan positions.
    overlap : float
        Overlap fraction used.
    photon_count : float
        Photon count used.

    Returns
    -------
    dict
        Dictionary containing all metrics and aligned reconstruction.
    """
    recon = recon_result['object']
    errors = recon_result['errors']
    gt = gt_object

    # Global complex scaling: a = <gt, recon> / <recon, recon>
    a = np.sum(gt * np.conj(recon)) / (np.sum(np.abs(recon)**2) + 1e-30)
    recon_a = recon * a

    gt_amp = np.abs(gt)
    rc_amp = np.abs(recon_a)

    # Per-pixel amplitude alignment via linear fit
    mask = gt_amp > 0.05 * gt_amp.max()
    if mask.sum() > 10:
        c = np.polyfit(rc_amp[mask].ravel(), gt_amp[mask].ravel(), 1)
        rc_amp_s = np.clip(c[0] * rc_amp + c[1], 0, None)
    else:
        rc_amp_s = rc_amp

    # Normalise to [0, 1] using GT range
    lo, hi = gt_amp.min(), gt_amp.max()
    gt_n = (gt_amp - lo) / (hi - lo + 1e-10)
    rc_n = np.clip((rc_amp_s - lo) / (hi - lo + 1e-10), 0, 1)

    # Compute amplitude metrics
    p = float(psnr(gt_n, rc_n, data_range=1.0))
    s = float(ssim(gt_n, rc_n, data_range=1.0))
    r = float(np.sqrt(np.mean((gt_n - rc_n)**2)))

    # Phase metrics
    gt_ph = np.angle(gt)
    rc_ph = np.angle(recon_a)
    if mask.sum() > 0:
        diff = np.angle(np.exp(1j * (rc_ph[mask] - gt_ph[mask])))
        offset = np.median(diff)
        rc_ph_c = rc_ph - offset
        diff2 = np.angle(np.exp(1j * (rc_ph_c[mask] - gt_ph[mask])))
        ph_err = float(np.sqrt(np.mean(diff2**2)))
        ph_corr = float(np.corrcoef(gt_ph[mask].ravel(),
                                     rc_ph_c[mask].ravel())[0, 1])
    else:
        ph_err, ph_corr = float('inf'), 0.0
        rc_ph_c = rc_ph

    # Complex correlation
    cc = float(np.abs(np.sum(recon_a * np.conj(gt))) /
               (np.sqrt(np.sum(np.abs(recon_a)**2) *
                        np.sum(np.abs(gt)**2)) + 1e-30))

    # Aligned reconstruction
    recon_aligned = rc_amp_s * np.exp(1j * rc_ph_c)

    metrics = {
        'psnr': p,
        'ssim': s,
        'rmse': r,
        'phase_error_rad': ph_err,
        'phase_correlation': ph_corr,
        'complex_correlation': cc
    }

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    # Save outputs
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), np.abs(gt))
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), np.abs(recon_aligned))
    np.save(os.path.join(RESULTS_DIR, "gt_complex.npy"), gt)
    np.save(os.path.join(RESULTS_DIR, "recon_complex.npy"), recon_aligned)

    metrics_out = {
        **metrics,
        'n_iterations': n_iter,
        'n_scan_positions': n_positions,
        'overlap': overlap,
        'photon_count': photon_count
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Generate visualization
    plot_results(gt, recon_aligned, metrics, errors,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    return {
        'metrics': metrics,
        'recon_aligned': recon_aligned,
        'errors': errors
    }
