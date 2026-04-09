import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data_dict, result_dict):
    """
    Evaluate inversion results and save outputs.
    
    Args:
        data_dict: Dictionary containing ground truth data
        result_dict: Dictionary containing inversion results
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    param_names = ["log10_A_gw", "log10_A_red", "gamma_red"]
    true_params = data_dict['true_params']
    medians = result_dict['medians']
    stds = result_dict['stds']
    samples = result_dict['samples']
    freqs = data_dict['freqs']
    
    psd_red_true = data_dict['psd_red_true']
    psd_gw_true = data_dict['psd_gw_true']
    psd_red_recon = result_dict['psd_red_recon']
    psd_gw_recon = result_dict['psd_gw_recon']
    
    # Print parameter recovery results
    print(f"\n  {'Parameter':<15s} {'True':>10s} {'Median':>10s} {'Std':>10s}")
    print("  " + "-" * 50)
    for i, name in enumerate(param_names):
        print(f"  {name:<15s} {true_params[i]:10.3f} {medians[i]:10.3f} {stds[i]:10.3f}")

    # Relative errors
    re_values = {}
    for i, name in enumerate(param_names):
        if abs(true_params[i]) > 1e-10:
            re = abs(medians[i] - true_params[i]) / abs(true_params[i])
        else:
            re = abs(medians[i] - true_params[i])
        re_values[name] = float(re)

    # Cross-correlation of log-PSDs (red noise)
    log_psd_true = np.log10(psd_red_true + 1e-100)
    log_psd_recon = np.log10(psd_red_recon + 1e-100)
    cc_num = np.sum((log_psd_true - log_psd_true.mean()) *
                     (log_psd_recon - log_psd_recon.mean()))
    cc_den = np.sqrt(np.sum((log_psd_true - log_psd_true.mean()) ** 2) *
                     np.sum((log_psd_recon - log_psd_recon.mean()) ** 2))
    psd_cc = float(cc_num / (cc_den + 1e-30))

    # Cross-correlation of log-PSDs (GW)
    log_gw_true = np.log10(psd_gw_true + 1e-100)
    log_gw_recon = np.log10(psd_gw_recon + 1e-100)
    cc_gw_num = np.sum((log_gw_true - log_gw_true.mean()) *
                        (log_gw_recon - log_gw_recon.mean()))
    cc_gw_den = np.sqrt(np.sum((log_gw_true - log_gw_true.mean()) ** 2) *
                        np.sum((log_gw_recon - log_gw_recon.mean()) ** 2))
    psd_gw_cc = float(cc_gw_num / (cc_gw_den + 1e-30))

    mean_re = float(np.mean(list(re_values.values())))

    print(f"\n  Mean relative error: {mean_re:.4f}")
    print(f"  Red noise PSD CC:   {psd_cc:.4f}")
    print(f"  GW PSD CC:          {psd_gw_cc:.4f}")

    # Build metrics dictionary
    metrics = {
        "log10_A_gw_true": float(true_params[0]),
        "log10_A_gw_recovered": float(medians[0]),
        "log10_A_gw_std": float(stds[0]),
        "log10_A_gw_RE": re_values["log10_A_gw"],
        "log10_A_red_true": float(true_params[1]),
        "log10_A_red_recovered": float(medians[1]),
        "log10_A_red_std": float(stds[1]),
        "log10_A_red_RE": re_values["log10_A_red"],
        "gamma_red_true": float(true_params[2]),
        "gamma_red_recovered": float(medians[2]),
        "gamma_red_std": float(stds[2]),
        "gamma_red_RE": re_values["gamma_red"],
        "mean_parameter_RE": mean_re,
        "red_noise_PSD_CC": psd_cc,
        "GW_PSD_CC": psd_gw_cc,
        "n_pulsars": data_dict['n_pulsars'],
        "n_toa": data_dict['n_toa'],
        "n_walkers": result_dict['n_walkers'],
        "n_steps": result_dict['n_steps'],
        "n_burn": result_dict['n_burn'],
    }

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  → metrics.json saved")

    # Save ground truth
    gt_dict = {
        "true_params": true_params,
        "freqs": freqs,
        "psd_red_true": psd_red_true,
        "psd_gw_true": psd_gw_true,
        "residuals": [r for r in data_dict['residuals_all']],
        "hd_matrix": data_dict['hd_matrix'],
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_dict,
            allow_pickle=True)

    # Save reconstruction
    recon_dict = {
        "recovered_params": medians,
        "param_stds": stds,
        "psd_red_recon": psd_red_recon,
        "psd_gw_recon": psd_gw_recon,
        "samples": samples,
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_dict,
            allow_pickle=True)
    print("  → ground_truth.npy, reconstruction.npy saved")

    # ── Visualization ─────────────────────────────────────────────────────
    chain = result_dict['chain']
    n_walkers = result_dict['n_walkers']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0), (0,1), (0,2) Trace plots
    for i, name in enumerate(param_names):
        ax = axes[0, i]
        for w in range(n_walkers):
            ax.plot(chain[:, w, i], alpha=0.3, lw=0.5)
        ax.axhline(true_params[i], color='r', lw=2, label='Truth')
        ax.axhline(medians[i], color='blue', ls='--', lw=1.5, label='Median')
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'Trace: {name}')
        ax.legend(fontsize=8)

    # (1,0) Red noise PSD
    ax = axes[1, 0]
    ax.loglog(freqs * 365.25 * 86400, psd_red_true, 'r-', lw=2, label='True red noise')
    ax.loglog(freqs * 365.25 * 86400, psd_red_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'Red Noise PSD (CC={psd_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) GW PSD
    ax = axes[1, 1]
    ax.loglog(freqs * 365.25 * 86400, psd_gw_true, 'r-', lw=2, label='True GWB')
    ax.loglog(freqs * 365.25 * 86400, psd_gw_recon, 'b--', lw=2, label='Recovered')
    ax.set_xlabel('Frequency (1/yr)')
    ax.set_ylabel('PSD (s²/Hz)')
    ax.set_title(f'GW Background PSD (CC={psd_gw_cc:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Corner-like: 2D posterior (A_gw vs A_red)
    ax = axes[1, 2]
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.2, c='steelblue')
    ax.axvline(true_params[0], color='r', lw=1.5, label='True')
    ax.axhline(true_params[1], color='r', lw=1.5)
    ax.scatter([medians[0]], [medians[1]], c='blue', marker='x', s=100,
               zorder=5, label='Median')
    ax.set_xlabel('log10_A_gw')
    ax.set_ylabel('log10_A_red')
    ax.set_title('Posterior: A_gw vs A_red')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {fig_path} saved")

    print("\n" + "=" * 60)
    print("DONE — PTA Bayesian inference complete")
    print(f"  Mean param RE = {mean_re:.4f}")
    print(f"  Red PSD CC    = {psd_cc:.4f}")
    print(f"  GW PSD CC     = {psd_gw_cc:.4f}")
    print("=" * 60)

    return metrics
