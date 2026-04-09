import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data, cl_pseudo, result):
    """
    Evaluate reconstruction quality: compute metrics, save results,
    and generate visualization.

    Parameters
    ----------
    data : dict from load_and_preprocess_data
    cl_pseudo : array from forward_operator
    result : dict from run_inversion

    Returns
    -------
    metrics : dict with PSNR, CC, RMSE, mean relative error
    """
    cl_true = data['cl_true']
    lmax = data['lmax']
    cl_recon = result['cl_recon']
    ell_eff = result['ell_eff']

    # Interpolate true Cl at the effective ell values of each bin
    ell_all = np.arange(len(cl_true))
    cl_true_binned = np.interp(ell_eff, ell_all, cl_true)

    # Keep only ell >= 2 (monopole/dipole undefined)
    valid = ell_eff >= 2
    t = cl_true_binned[valid]
    r = cl_recon[valid]

    # PSNR
    data_range = np.max(t) - np.min(t)
    mse = np.mean((t - r) ** 2)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')

    # Pearson CC
    cc = float(np.corrcoef(t, r)[0, 1])

    # Relative error
    re = float(np.mean(np.abs(t - r) / (np.abs(t) + 1e-30)))

    # RMSE
    rmse = float(np.sqrt(mse))

    metrics = {
        "psnr_dB": float(psnr),
        "correlation_coefficient": cc,
        "rmse": rmse,
        "mean_relative_error": re,
        "method": "NaMaster_pseudo_Cl_deconvolution",
    }

    # Print metrics
    print(f"[EVAL] PSNR  = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC    = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] RMSE  = {metrics['rmse']:.6e}")
    print(f"[EVAL] RE    = {metrics['mean_relative_error']:.4f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # Visualization: four-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cl_true_at_ell = np.interp(ell_eff, ell_all, cl_true)

    def D(ell_arr, cl_arr):
        return ell_arr * (ell_arr + 1) * cl_arr / (2 * np.pi)

    # (a) True power spectrum
    ax = axes[0, 0]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b-', lw=1.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(a) True Power Spectrum')
    ax.set_xlim([2, lmax])

    # (b) Pseudo-Cl (naive) vs True
    ax = axes[0, 1]
    ax.plot(ell_all[2:], D(ell_all[2:], cl_pseudo[2:]), 'r-', alpha=0.7, lw=1, label='Pseudo-Cℓ')
    ax.plot(ell_all[2:], D(ell_all[2:], cl_true[2:]), 'b--', alpha=0.5, lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(b) Pseudo-Cℓ (biased) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # (c) Decoupled Cl (NaMaster) vs True
    ax = axes[1, 0]
    ax.plot(ell_eff, D(ell_eff, cl_recon), 'go-', ms=3, lw=1.5, label='NaMaster')
    ax.plot(ell_eff, D(ell_eff, cl_true_at_ell), 'b--', lw=1, label='True')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ℓ(ℓ+1)Cℓ / 2π')
    ax.set_title('(c) Decoupled Cℓ (NaMaster) vs True')
    ax.legend()
    ax.set_xlim([2, lmax])

    # (d) Relative error per bin
    ax = axes[1, 1]
    rel_err = np.abs(cl_recon[valid] - cl_true_at_ell[valid]) / (np.abs(cl_true_at_ell[valid]) + 1e-30)
    ax.semilogy(ell_eff[valid], rel_err, 'k.-', ms=3)
    ax.axhline(y=0.1, color='r', ls='--', alpha=0.5, label='10% error')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('|Cℓ_recon − Cℓ_true| / Cℓ_true')
    ax.set_title('(d) Relative Error per ℓ-bin')
    ax.legend()
    ax.set_xlim([2, lmax])

    fig.suptitle(
        f"NaMaster Pseudo-Cℓ Deconvolution  |  "
        f"PSNR={metrics['psnr_dB']:.2f} dB  |  "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {vis_path}")

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), cl_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), cl_recon)
    np.save(os.path.join(RESULTS_DIR, "observed_data.npy"), cl_pseudo)
    np.save(os.path.join(RESULTS_DIR, "ell_effective.npy"), ell_eff)
    print("[SAVE] Arrays saved (ground_truth, recon_output, observed_data, ell_effective)")

    return metrics
