import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(k, chi_meas, chi_clean, chi_fit, gt_shells, fit_shells,
                     k_weight, results_dir):
    """
    Compute metrics, save outputs, and generate visualization.
    
    Parameters
    ----------
    k : ndarray
        Photoelectron wavenumber array [Å^-1].
    chi_meas : ndarray
        Measured (noisy) EXAFS oscillation.
    chi_clean : ndarray
        Clean (ground truth) EXAFS oscillation.
    chi_fit : ndarray
        Fitted EXAFS oscillation.
    gt_shells : list of dict
        Ground truth shell parameters.
    fit_shells : list of dict
        Fitted shell parameters.
    k_weight : int
        k-weighting exponent.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    # k-weighted χ
    kw_gt = chi_clean * k**k_weight
    kw_fit = chi_fit * k**k_weight
    
    # Correlation coefficient
    cc = float(np.corrcoef(kw_gt, kw_fit)[0, 1])
    
    # RMSE
    rmse = float(np.sqrt(np.mean((kw_gt - kw_fit)**2)))
    
    # Data range and MSE
    dr = kw_gt.max() - kw_gt.min()
    mse = np.mean((kw_gt - kw_fit)**2)
    
    # PSNR
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    
    # 1-D SSIM: tile to make 2D (7×N) so win_size=7 works
    tile_rows = 7
    a2d = np.tile(kw_gt, (tile_rows, 1))
    b2d = np.tile(kw_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))
    
    # Relative error
    re = float(np.linalg.norm(kw_gt - kw_fit) / max(np.linalg.norm(kw_gt), 1e-12))
    
    # R-space (FT) comparison
    window = np.hanning(len(k))
    ft_gt = np.abs(np.fft.fft(kw_gt * window))[:len(k)//2]
    ft_fit = np.abs(np.fft.fft(kw_fit * window))[:len(k)//2]
    cc_ft = float(np.corrcoef(ft_gt, ft_fit)[0, 1])
    
    # Parameter recovery metrics
    param_metrics = {}
    for i, (gt_sh, fit_sh) in enumerate(zip(gt_shells, fit_shells)):
        for key in ["N", "R", "sigma2"]:
            g, f = gt_sh[key], fit_sh[key]
            param_metrics[f"gt_{gt_sh['label']}_{key}"] = float(g)
            param_metrics[f"fit_{gt_sh['label']}_{key}"] = float(f)
            param_metrics[f"err_{gt_sh['label']}_{key}"] = float(abs(g - f))
    
    metrics = {
        "PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse,
        "CC_FT": cc_ft, **param_metrics
    }
    
    # Print metrics
    for key, val in sorted(metrics.items()):
        print(f"  {key:30s} = {val}")
    
    # Save metrics to JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), chi_fit)
    np.save(os.path.join(results_dir, "ground_truth.npy"), chi_clean)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) k²χ(k)
    axes[0, 0].plot(k, chi_clean * k**2, 'b-', lw=2, label='GT')
    axes[0, 0].plot(k, chi_meas * k**2, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0, 0].plot(k, chi_fit * k**2, 'r--', lw=1.5, label='Fit')
    axes[0, 0].set_xlabel('k [Å⁻¹]')
    axes[0, 0].set_ylabel('k²χ(k)')
    axes[0, 0].set_title('(a) EXAFS')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Fourier transform (R-space)
    r = np.fft.fftfreq(len(k), d=(k[1] - k[0]) / (2 * np.pi))[:len(k)//2]
    axes[0, 1].plot(r, ft_gt, 'b-', lw=2, label='GT')
    axes[0, 1].plot(r, ft_fit, 'r--', lw=1.5, label='Fit')
    axes[0, 1].set_xlabel('R [Å]')
    axes[0, 1].set_ylabel('|FT|')
    axes[0, 1].set_title('(b) Radial Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 5)
    
    # (c) Residual
    axes[1, 0].plot(k, (chi_clean - chi_fit) * k**2, 'g-', lw=1)
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('k [Å⁻¹]')
    axes[1, 0].set_ylabel('Residual k²Δχ')
    axes[1, 0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (d) Parameter bars
    labels, gt_v, fit_v = [], [], []
    for gs, fs in zip(gt_shells, fit_shells):
        for key in ["N", "R", "sigma2"]:
            labels.append(f"{gs['label']}_{key}")
            gt_v.append(gs[key])
            fit_v.append(fs[key])
    x = np.arange(len(labels))
    w = 0.35
    axes[1, 1].bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    axes[1, 1].bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=7, rotation=30)
    axes[1, 1].set_title('(d) Parameters')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"xraylarch — EXAFS Fitting\nPSNR={metrics['PSNR']:.1f} dB  |  "
                 f"SSIM={metrics['SSIM']:.4f}  |  CC={metrics['CC']:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
