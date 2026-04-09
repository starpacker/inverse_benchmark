import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_113_diffpy_pdf"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def evaluate_results(r, G_gt, G_noisy, G_fit, params_true, a_fit, B_fit, scale_fit):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, correlation coefficient, parameter errors, generates
    visualization plots, and saves all outputs to disk.
    
    Parameters:
    -----------
    r : ndarray
        r grid values
    G_gt : ndarray
        Ground truth G(r)
    G_noisy : ndarray
        Noisy measured G(r)
    G_fit : ndarray
        Fitted G(r)
    params_true : dict
        Dictionary containing true parameters ('a', 'B', 'scale')
    a_fit : float
        Fitted lattice constant
    B_fit : float
        Fitted Debye-Waller factor
    scale_fit : float
        Fitted scale factor
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics
    """
    a_true = params_true['a']
    B_true = params_true['B']
    scale_true = params_true['scale']
    
    g_max = np.max(np.abs(G_gt)) + 1e-12
    gt_n = G_gt / g_max
    fi_n = G_fit / g_max
    
    mse = np.mean((gt_n - fi_n)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    
    g = gt_n - np.mean(gt_n)
    f = fi_n - np.mean(fi_n)
    cc = np.sum(g * f) / (np.sqrt(np.sum(g**2) * np.sum(f**2)) + 1e-12)
    
    a_err = abs(a_fit - a_true) / a_true * 100
    B_err = abs(B_fit - B_true) / B_true * 100
    scale_err = abs(scale_fit - scale_true) / scale_true * 100
    
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "lattice_constant_true": float(a_true),
        "lattice_constant_fitted": float(a_fit),
        "lattice_constant_error_pct": float(a_err),
        "debye_waller_true": float(B_true),
        "debye_waller_fitted": float(B_fit),
        "debye_waller_error_pct": float(B_err),
        "scale_true": float(scale_true),
        "scale_fitted": float(scale_fit),
        "scale_error_pct": float(scale_err),
        "RMSE": float(np.sqrt(mse)),
    }
    
    r_min = r[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(r, G_gt, "b-", linewidth=1.5, label="Ground Truth G(r)")
    axes[0, 0].plot(r, G_fit, "r--", linewidth=1.5, label="Fitted G(r)")
    axes[0, 0].set_xlabel("r (Å)", fontsize=12)
    axes[0, 0].set_ylabel("G(r)", fontsize=12)
    axes[0, 0].set_title("PDF: Ground Truth vs Fit", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([r_min, 15])
    
    axes[0, 1].plot(r, G_noisy, "gray", alpha=0.6, linewidth=0.8, label="Noisy data")
    axes[0, 1].plot(r, G_fit, "r-", linewidth=1.5, label="Fitted G(r)")
    axes[0, 1].set_xlabel("r (Å)", fontsize=12)
    axes[0, 1].set_ylabel("G(r)", fontsize=12)
    axes[0, 1].set_title("Noisy Data vs Fit", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([r_min, 15])
    
    residual = G_gt - G_fit
    axes[1, 0].plot(r, residual, "g-", linewidth=1.0)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("r (Å)", fontsize=12)
    axes[1, 0].set_ylabel("Residual", fontsize=12)
    axes[1, 0].set_title(f"Residual (RMSE = {metrics['RMSE']:.4f})", fontsize=14)
    axes[1, 0].set_xlim([r_min, 15])
    
    axes[1, 1].axis("off")
    table_data = [
        ["Parameter", "True", "Fitted", "Error (%)"],
        ["a (Å)", f"{metrics['lattice_constant_true']:.4f}",
         f"{metrics['lattice_constant_fitted']:.4f}",
         f"{metrics['lattice_constant_error_pct']:.2f}%"],
        ["B (Å²)", f"{metrics['debye_waller_true']:.4f}",
         f"{metrics['debye_waller_fitted']:.4f}",
         f"{metrics['debye_waller_error_pct']:.2f}%"],
        ["Scale", f"{metrics['scale_true']:.4f}",
         f"{metrics['scale_fitted']:.4f}",
         f"{metrics['scale_error_pct']:.2f}%"],
        ["", "", "", ""],
        ["PSNR", f"{metrics['PSNR']:.2f} dB", "", ""],
        ["CC", f"{metrics['CC']:.4f}", "", ""],
    ]
    table = axes[1, 1].table(
        cellText=table_data, loc="center", cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Fitted Parameters", fontsize=14)
    
    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), G_gt)
        np.save(os.path.join(d, "recon_output.npy"), G_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
