import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from scipy.signal import find_peaks

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(result_dict, results_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary containing:
        - 'freq': frequency array
        - 'tau': relaxation time array
        - 'gamma_gt': ground truth DRT
        - 'gamma_rec': recovered DRT
        - 'Z_clean': clean impedance
        - 'Z_noisy': noisy impedance
        - 'Z_fit': fitted impedance
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    freq = result_dict['freq']
    tau = result_dict['tau']
    gamma_gt = result_dict['gamma_gt']
    gamma_rec = result_dict['gamma_rec']
    Z_clean = result_dict['Z_clean']
    Z_noisy = result_dict['Z_noisy']
    Z_fit = result_dict['Z_fit']
    
    print("\n[EVAL] Computing metrics ...")
    
    # DRT metrics (normalized)
    g_gt = gamma_gt / max(gamma_gt.max(), 1e-12)
    g_rec = gamma_rec / max(gamma_rec.max(), 1e-12)
    
    cc_drt = float(np.corrcoef(g_gt, g_rec)[0, 1])
    re_drt = float(np.linalg.norm(g_gt - g_rec) / max(np.linalg.norm(g_gt), 1e-12))
    rmse_drt = float(np.sqrt(np.mean((g_gt - g_rec) ** 2)))
    
    data_range = g_gt.max() - g_gt.min()
    mse = np.mean((g_gt - g_rec) ** 2)
    psnr_drt = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    
    tile_rows = 7
    a2d = np.tile(g_gt, (tile_rows, 1))
    b2d = np.tile(g_rec, (tile_rows, 1))
    ssim_drt = float(ssim_fn(a2d, b2d, data_range=data_range, win_size=7))
    
    # Impedance fit metrics
    Z_resid = Z_clean - Z_fit
    rmse_Z = float(np.sqrt(np.mean(np.abs(Z_resid) ** 2)))
    cc_Z_re = float(np.corrcoef(Z_clean.real, Z_fit.real)[0, 1])
    cc_Z_im = float(np.corrcoef(Z_clean.imag, Z_fit.imag)[0, 1])
    
    # Peak detection
    peaks_gt, _ = find_peaks(g_gt, height=0.1)
    peaks_rec, _ = find_peaks(g_rec, height=0.1)
    
    metrics = {
        "PSNR_DRT": psnr_drt,
        "SSIM_DRT": ssim_drt,
        "CC_DRT": cc_drt,
        "RE_DRT": re_drt,
        "RMSE_DRT": rmse_drt,
        "CC_Z_real": cc_Z_re,
        "CC_Z_imag": cc_Z_im,
        "RMSE_Z": rmse_Z,
        "n_peaks_gt": len(peaks_gt),
        "n_peaks_rec": len(peaks_rec),
    }
    
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics and data
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), gamma_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gamma_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) DRT
    ax = axes[0, 0]
    ax.semilogx(tau, gamma_gt / max(gamma_gt.max(), 1e-12),
                'b-', lw=2, label='GT')
    ax.semilogx(tau, gamma_rec / max(gamma_rec.max(), 1e-12),
                'r--', lw=2, label='Recovered')
    ax.set_xlabel('τ [s]')
    ax.set_ylabel('γ(τ) [normalised]')
    ax.set_title('(a) Distribution of Relaxation Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Nyquist plot
    ax = axes[0, 1]
    ax.plot(Z_clean.real, -Z_clean.imag, 'b-', lw=2, label='GT')
    ax.plot(Z_noisy.real, -Z_noisy.imag, 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.plot(Z_fit.real, -Z_fit.imag, 'r--', lw=2, label='Fit')
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title('(b) Nyquist Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # (c) Bode magnitude
    ax = axes[1, 0]
    ax.loglog(freq, np.abs(Z_clean), 'b-', lw=2, label='GT')
    ax.loglog(freq, np.abs(Z_noisy), 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.loglog(freq, np.abs(Z_fit), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|Z| [Ω]')
    ax.set_title('(c) Bode Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # (d) Bode phase
    ax = axes[1, 1]
    ax.semilogx(freq, np.degrees(np.angle(Z_clean)), 'b-', lw=2, label='GT')
    ax.semilogx(freq, np.degrees(np.angle(Z_noisy)), 'k.', ms=3, alpha=0.5)
    ax.semilogx(freq, np.degrees(np.angle(Z_fit)), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [°]')
    ax.set_title('(d) Bode Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pyDRTtools — DRT Inversion from EIS\n"
        f"PSNR={metrics['PSNR_DRT']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_DRT']:.4f}  |  CC={metrics['CC_DRT']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
