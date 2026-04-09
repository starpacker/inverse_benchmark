import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(true_params, fitted_params,
                     G_prime_true, G_double_prime_true,
                     G_prime_fit, G_double_prime_fit,
                     omega, G_prime_obs, G_double_prime_obs,
                     results_dir):
    """
    Evaluate inversion results: compute metrics, save outputs, and generate visualizations.

    Parameters
    ----------
    true_params : dict
        Ground truth parameters.
    fitted_params : dict
        Recovered parameters from inversion.
    G_prime_true : ndarray
        True storage modulus (noise-free).
    G_double_prime_true : ndarray
        True loss modulus (noise-free).
    G_prime_fit : ndarray
        Fitted storage modulus.
    G_double_prime_fit : ndarray
        Fitted loss modulus.
    omega : ndarray
        Angular frequencies.
    G_prime_obs : ndarray
        Observed storage modulus (with noise).
    G_double_prime_obs : ndarray
        Observed loss modulus (with noise).
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Compute per-parameter relative errors ──
    param_errors = {}
    for key in ('G0', 'tau_R', 'eta_s'):
        tv = true_params[key]
        fv = fitted_params[key]
        re = abs(tv - fv) / abs(tv)
        param_errors[key] = {'true': float(tv), 'fitted': float(fv), 'rel_error': float(re)}

    mean_re = float(np.mean([v['rel_error'] for v in param_errors.values()]))

    # ── Concatenate G' and G'' for spectral metrics (log scale) ──
    EPS = 1e-30
    log_true = np.log10(np.concatenate([G_prime_true, G_double_prime_true]) + EPS)
    log_fit = np.log10(np.concatenate([G_prime_fit, G_double_prime_fit]) + EPS)

    data_range = float(log_true.max() - log_true.min())
    mse = float(np.mean((log_true - log_fit) ** 2))
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')

    cc = float(np.corrcoef(log_true, log_fit)[0, 1])

    metrics = {
        'psnr_dB': float(psnr),
        'correlation_coefficient': cc,
        'mean_parameter_relative_error': mean_re,
        'parameters': param_errors,
        'method': 'Rouse_model_differential_evolution_fitting',
    }

    # ── Print metrics ──
    print(f"[EVAL] PSNR = {metrics['psnr_dB']:.2f} dB")
    print(f"[EVAL] CC   = {metrics['correlation_coefficient']:.6f}")
    print(f"[EVAL] Mean RE = {metrics['mean_parameter_relative_error']:.6f}")
    for k, v in metrics['parameters'].items():
        print(f"       {k:>6s}: true={v['true']:.4e}  fitted={v['fitted']:.4e}  RE={v['rel_error']:.6f}")

    # ── Save metrics ──
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # ── Save arrays ──
    np.save(os.path.join(results_dir, "ground_truth.npy"),
            np.column_stack([G_prime_true, G_double_prime_true]))
    np.save(os.path.join(results_dir, "recon_output.npy"),
            np.column_stack([G_prime_fit, G_double_prime_fit]))
    print(f"[SAVE] ground_truth.npy, recon_output.npy → {results_dir}")

    # ── Visualize ──
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Storage modulus G'
    ax = axes[0, 0]
    ax.loglog(omega, G_prime_true, 'b-', lw=2, label="G' (true)")
    ax.loglog(omega, G_prime_obs, 'rx', ms=4, alpha=0.5, label="G' (observed)")
    ax.loglog(omega, G_prime_fit, 'g--', lw=2, label="G' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G' (Pa)", fontsize=11)
    ax.set_title("(a) Storage Modulus G'", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (b) Loss modulus G''
    ax = axes[0, 1]
    ax.loglog(omega, G_double_prime_true, 'b-', lw=2, label="G'' (true)")
    ax.loglog(omega, G_double_prime_obs, 'rx', ms=4, alpha=0.5, label="G'' (observed)")
    ax.loglog(omega, G_double_prime_fit, 'g--', lw=2, label="G'' (fitted)")
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel("G'' (Pa)", fontsize=11)
    ax.set_title("(b) Loss Modulus G''", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # (c) Parameter comparison (bar chart, log scale)
    ax = axes[1, 0]
    params = metrics['parameters']
    names = list(params.keys())
    true_vals = [params[n]['true'] for n in names]
    fit_vals = [params[n]['fitted'] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    ax.bar(x_pos - w / 2, true_vals, w, label='True', color='steelblue', alpha=0.8)
    ax.bar(x_pos + w / 2, fit_vals, w, label='Fitted', color='seagreen', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(c) Parameter Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # (d) Residuals (% relative error vs true, noise-free)
    ax = axes[1, 1]
    res_p = (G_prime_fit - G_prime_true) / G_prime_true * 100.0
    res_pp = (G_double_prime_fit - G_double_prime_true) / G_double_prime_true * 100.0
    ax.semilogx(omega, res_p, 'b.-', ms=3, label="G' residual")
    ax.semilogx(omega, res_pp, 'r.-', ms=3, label="G'' residual")
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('(d) Fit Residuals vs True (noise-free)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Polymer Rheology Inversion (Rouse Model)  |  "
        f"PSNR = {metrics['psnr_dB']:.2f} dB  |  "
        f"CC = {metrics['correlation_coefficient']:.4f}  |  "
        f"Mean RE = {metrics['mean_parameter_relative_error']:.4f}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS]  Saved → {vis_path}")

    return metrics
