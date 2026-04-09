import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def cauchy_n(wavelength_nm, A, B, C):
    """Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴"""
    lam_um = wavelength_nm / 1000.0
    return A + B / lam_um**2 + C / lam_um**4

def evaluate_results(data, inversion_result, results_dir):
    """
    Compute metrics, save outputs, and visualize results.

    Parameters
    ----------
    data : dict
        Dictionary containing wavelengths, clean data, gt_params, etc.
    inversion_result : dict
        Dictionary containing fit_params, psi_fit, delta_fit.
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    wavelengths = data["wavelengths"]
    psi_clean = data["psi_clean"]
    delta_clean = data["delta_clean"]
    psi_meas = data["psi_noisy"]
    delta_meas = data["delta_noisy"]
    gt = data["gt_params"]

    fit = inversion_result["fit_params"]
    psi_fit = inversion_result["psi_fit"]
    delta_fit = inversion_result["delta_fit"]

    # Ψ metrics
    rmse_psi = float(np.sqrt(np.mean((psi_clean - psi_fit)**2)))
    cc_psi = float(np.corrcoef(psi_clean, psi_fit)[0, 1])

    # Δ metrics
    rmse_delta = float(np.sqrt(np.mean((delta_clean - delta_fit)**2)))
    cc_delta = float(np.corrcoef(delta_clean, delta_fit)[0, 1])

    # Combined PSNR/SSIM on Ψ
    dr = psi_clean.max() - psi_clean.min()
    mse = np.mean((psi_clean - psi_fit)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    tile_rows = 7
    a2d = np.tile(psi_clean, (tile_rows, 1))
    b2d = np.tile(psi_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))

    # Parameter recovery
    param_metrics = {}
    for k in ["thickness", "A", "B", "C", "k_amp"]:
        g, f = gt[k], fit[k]
        param_metrics[f"gt_{k}"] = float(g)
        param_metrics[f"fit_{k}"] = float(f)
        param_metrics[f"abs_err_{k}"] = float(abs(g - f))

    # n(λ) recovery
    n_gt = cauchy_n(wavelengths, gt["A"], gt["B"], gt["C"])
    n_fit = cauchy_n(wavelengths, fit["A"], fit["B"], fit["C"])
    cc_n = float(np.corrcoef(n_gt, n_fit)[0, 1])

    metrics = {
        "PSNR_psi": psnr,
        "SSIM_psi": ssim_val,
        "CC_psi": cc_psi,
        "RMSE_psi_deg": rmse_psi,
        "CC_delta": cc_delta,
        "RMSE_delta_deg": rmse_delta,
        "CC_n": cc_n,
        **param_metrics,
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reconstructions
    np.save(
        os.path.join(results_dir, "reconstruction.npy"),
        np.column_stack([psi_fit, delta_fit])
    )
    np.save(
        os.path.join(results_dir, "ground_truth.npy"),
        np.column_stack([psi_clean, delta_clean])
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(wavelengths, psi_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, psi_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, psi_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Ψ [°]')
    ax.set_title('(a) Ψ(λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(wavelengths, delta_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, delta_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, delta_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Δ [°]')
    ax.set_title('(b) Δ(λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(wavelengths, n_gt, 'b-', lw=2, label='GT n(λ)')
    ax.plot(wavelengths, n_fit, 'r--', lw=2, label='Fit n(λ)')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Refractive index n')
    ax.set_title('(c) Dispersion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ['d [nm]', 'A', 'B', 'C', 'k_amp']
    keys = ['thickness', 'A', 'B', 'C', 'k_amp']
    gt_v = [gt[k] for k in keys]
    fit_v = [fit[k] for k in keys]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('(d) Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"refellips — Spectroscopic Ellipsometry Inversion\n"
        f"PSNR(Ψ)={metrics['PSNR_psi']:.1f} dB  |  CC(Ψ)={metrics['CC_psi']:.4f}  |  "
        f"CC(Δ)={metrics['CC_delta']:.4f}  |  Δd={metrics['abs_err_thickness']:.2f} nm",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics
