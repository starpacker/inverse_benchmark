import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data_dict, result_dict):
    """
    Compute MT inversion metrics and generate visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data.
    result_dict : dict
        Dictionary from run_inversion.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    # Extract data
    frequencies = data_dict['frequencies']
    rho_clean = data_dict['rho_clean']
    rho_noisy = data_dict['rho_noisy']
    phi_clean = data_dict['phi_clean']
    phi_noisy = data_dict['phi_noisy']
    gt_thick = data_dict['gt_thicknesses']
    gt_res = data_dict['gt_resistivities']
    
    rec_res = result_dict['rec_resistivities']
    depths = result_dict['depths']
    rho_rec = result_dict['rho_pred']
    phi_rec = result_dict['phi_pred']
    
    # Compute metrics
    print("\n[EVAL] Computing metrics ...")
    
    # Apparent resistivity curve metrics
    log_rho_gt = np.log10(rho_clean)
    log_rho_rec = np.log10(rho_rec)

    rmse_log_rho = float(np.sqrt(np.mean((log_rho_gt - log_rho_rec) ** 2)))
    cc_log_rho = float(np.corrcoef(log_rho_gt, log_rho_rec)[0, 1])

    # Phase metrics
    rmse_phi = float(np.sqrt(np.mean((phi_clean - phi_rec) ** 2)))
    cc_phi = float(np.corrcoef(phi_clean, phi_rec)[0, 1])

    # Combined PSNR
    data_range = log_rho_gt.max() - log_rho_gt.min()
    mse = np.mean((log_rho_gt - log_rho_rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    metrics = {
        "PSNR_logRho": psnr,
        "CC_logRho": cc_log_rho,
        "RMSE_logRho": rmse_log_rho,
        "CC_phase": cc_phi,
        "RMSE_phase_deg": rmse_phi,
        "GT_res_top": float(gt_res[0]),
        "GT_res_mid": float(gt_res[1]),
        "GT_res_bot": float(gt_res[2]),
        "min_recovered_res": float(rec_res.min()),
    }
    
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s} = {v}")

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save apparent resistivity curves
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rho_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), rho_clean)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Apparent resistivity sounding curve
    ax = axes[0, 0]
    ax.loglog(1/frequencies, rho_clean, 'b-', lw=2, label='GT')
    ax.loglog(1/frequencies, rho_noisy, 'k.', ms=5, alpha=0.5, label='Noisy')
    ax.loglog(1/frequencies, rho_rec, 'r--', lw=2, label='Fit')
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('ρ_a [Ω·m]')
    ax.set_title('(a) Apparent Resistivity')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (b) Phase
    ax = axes[0, 1]
    ax.semilogx(1/frequencies, phi_clean, 'b-', lw=2, label='GT')
    ax.semilogx(1/frequencies, phi_noisy, 'k.', ms=5, alpha=0.5, label='Noisy')
    ax.semilogx(1/frequencies, phi_rec, 'r--', lw=2, label='Fit')
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('Phase [°]')
    ax.set_title('(b) Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Resistivity-depth profile (step plot)
    ax = axes[1, 0]
    gt_depths = [0] + list(np.cumsum(gt_thick)) + [depths[-1] * 1.5]
    gt_res_plot = []
    for r in gt_res:
        gt_res_plot.extend([r, r])
    gt_d_plot = []
    for i in range(len(gt_depths) - 1):
        gt_d_plot.extend([gt_depths[i], gt_depths[i + 1]])

    ax.semilogx(gt_res_plot, gt_d_plot, 'b-', lw=2, label='GT')
    ax.semilogx(rec_res, [0] + list(depths), 'r.-', lw=1.5, label='Inversion')
    ax.invert_yaxis()
    ax.set_xlabel('Resistivity [Ω·m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title('(c) Resistivity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (d) Data fit
    ax = axes[1, 1]
    ax.plot(np.log10(rho_clean), np.log10(rho_rec), 'b.', ms=8)
    lims = [min(np.log10(rho_clean).min(), np.log10(rho_rec).min()),
            max(np.log10(rho_clean).max(), np.log10(rho_rec).max())]
    ax.plot(lims, lims, 'k--', lw=0.5)
    ax.set_xlabel('log₁₀(ρ_a GT) [Ω·m]')
    ax.set_ylabel('log₁₀(ρ_a fit) [Ω·m]')
    ax.set_title(f'(d) Data Fit  CC={metrics["CC_logRho"]:.4f}')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SimPEG — 1D Magnetotelluric Inversion\n"
        f"PSNR(logρ)={metrics['PSNR_logRho']:.1f} dB  |  "
        f"CC(logρ)={metrics['CC_logRho']:.4f}  |  "
        f"RMSE(φ)={metrics['RMSE_phase_deg']:.2f}°",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
