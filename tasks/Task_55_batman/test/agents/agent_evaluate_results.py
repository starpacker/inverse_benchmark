import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(gt_params, fit_params, flux_clean, flux_fit, flux_meas,
                     t, results_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters
    ----------
    gt_params : dict
        Ground-truth parameters.
    fit_params : dict
        Fitted parameters.
    flux_clean : np.ndarray
        Ground-truth flux.
    flux_fit : np.ndarray
        Fitted flux.
    flux_meas : np.ndarray
        Measured (noisy) flux.
    t : np.ndarray
        Time array.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    print("\n[EVAL] Computing metrics ...")
    
    # Light-curve metrics
    residual = flux_clean - flux_fit
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    cc = float(np.corrcoef(flux_clean, flux_fit)[0, 1])

    data_range = flux_clean.max() - flux_clean.min()
    mse = np.mean(residual ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    tile_rows = 7
    a2d = np.tile(flux_clean, (tile_rows, 1))
    b2d = np.tile(flux_fit, (tile_rows, 1))
    ssim = float(ssim_fn(
        a2d, b2d,
        data_range=data_range, win_size=7
    ))

    # Relative error
    norm_gt = np.linalg.norm(flux_clean)
    re = float(np.linalg.norm(residual) / max(norm_gt, 1e-12))

    # Parameter recovery
    free_keys = ["rp", "a", "inc", "u1", "u2"]
    param_metrics = {}
    for k in free_keys:
        gt_v = gt_params[k]
        fit_v = fit_params[k]
        param_metrics[f"gt_{k}"] = float(gt_v)
        param_metrics[f"fit_{k}"] = float(fit_v)
        param_metrics[f"abs_err_{k}"] = float(abs(gt_v - fit_v))
        if abs(gt_v) > 1e-12:
            param_metrics[f"rel_err_{k}_pct"] = float(abs(gt_v - fit_v) / abs(gt_v) * 100)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim,
        "CC": cc,
        "RMSE": rmse,
        "RE": re,
        **param_metrics,
    }
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), flux_fit)
    np.save(os.path.join(results_dir, "ground_truth.npy"), flux_clean)
    np.save(os.path.join(results_dir, "measurements.npy"), flux_meas)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Light curves
    ax = axes[0, 0]
    ax.plot(t * 24, flux_meas, 'k.', ms=1, alpha=0.3, label='Noisy data')
    ax.plot(t * 24, flux_clean, 'b-', lw=2, label='Ground truth')
    ax.plot(t * 24, flux_fit, 'r--', lw=1.5, label='batman fit')
    ax.set_xlabel('Time from mid-transit [hours]')
    ax.set_ylabel('Relative flux')
    ax.set_title('(a) Transit Light Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residuals
    ax = axes[0, 1]
    residual_ppm = (flux_clean - flux_fit) * 1e6
    ax.plot(t * 24, residual_ppm, 'g-', lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Residual [ppm]')
    ax.set_title(f'(b) Residuals  RMSE={metrics["RMSE"]*1e6:.1f} ppm')
    ax.grid(True, alpha=0.3)

    # (c) Transit depth zoom
    ax = axes[1, 0]
    mask = np.abs(t * 24) < 2  # within ±2 hours
    ax.plot(t[mask] * 24, flux_clean[mask], 'b-', lw=2, label='GT')
    ax.plot(t[mask] * 24, flux_fit[mask], 'r--', lw=2, label='Fit')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Flux')
    ax.set_title('(c) Transit Detail (±2 hr)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Parameter bar chart
    ax = axes[1, 1]
    keys = ["rp", "a", "inc", "u1", "u2"]
    labels = ["Rp/Rs", "a/Rs", "inc [°]", "u₁", "u₂"]
    gt_vals = [gt_params[k] for k in keys]
    fit_vals = [fit_params[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_vals, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"batman — Transit Photometry Inversion\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RMSE={metrics['RMSE']*1e6:.1f} ppm",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
