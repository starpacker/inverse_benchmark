import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(x, measured_spectrum, fitted_spectrum, clean_with_baseline, 
                     true_peaks, result, results_dir, fitted_baseline, individual_peaks_fit,
                     snr_target):
    """
    Evaluate fitting results and save outputs.
    
    Args:
        x: numpy array of x-axis values
        measured_spectrum: numpy array of measured spectrum
        fitted_spectrum: numpy array of fitted spectrum
        clean_with_baseline: numpy array of ground truth clean spectrum
        true_peaks: list of dicts with ground-truth peak parameters
        result: lmfit MinimizerResult object
        results_dir: path to results directory
        fitted_baseline: numpy array of fitted baseline
        individual_peaks_fit: list of numpy arrays for individual fitted peaks
        snr_target: target SNR in dB
    
    Returns:
        metrics: dict of evaluation metrics
    """
    print("[EVAL] Computing evaluation metrics ...")
    
    gt_signal = clean_with_baseline
    
    # Parameter-level relative errors
    param_errors = {}
    for i, pk in enumerate(true_peaks):
        pe = {}
        amp_fit = result.params[f'p{i}_amplitude'].value
        cen_fit = result.params[f'p{i}_center'].value
        pe['amplitude_RE'] = abs(amp_fit - pk['amplitude']) / pk['amplitude']
        pe['center_RE'] = abs(cen_fit - pk['center']) / pk['center']
        
        if pk["type"] in ("gaussian", "voigt"):
            sig_fit = result.params[f'p{i}_sigma'].value
            pe['sigma_RE'] = abs(sig_fit - pk['sigma']) / pk['sigma']
        
        if pk["type"] in ("lorentzian", "voigt"):
            gam_fit = result.params[f'p{i}_gamma'].value
            pe['gamma_RE'] = abs(gam_fit - pk['gamma']) / pk['gamma']
        
        pe['type'] = pk['type']
        pe['true_center'] = pk['center']
        pe['fitted_center'] = float(cen_fit)
        pe['true_amplitude'] = pk['amplitude']
        pe['fitted_amplitude'] = float(amp_fit)
        param_errors[f'peak_{i}'] = pe
        avg_re = np.mean([v for k, v in pe.items() if k.endswith('_RE')])
        print(f"[EVAL]   Peak {i} ({pk['type']}, center={pk['center']}): avg RE = {avg_re:.4f}")
    
    # Curve-level metrics
    mse = np.mean((gt_signal - fitted_spectrum) ** 2)
    data_range = np.max(gt_signal) - np.min(gt_signal)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    cc = np.corrcoef(gt_signal, fitted_spectrum)[0, 1]
    rmse = np.sqrt(mse)
    
    residuals = measured_spectrum - fitted_spectrum
    res_std = np.std(residuals)
    
    # Mean parameter RE across all peaks
    all_re = []
    for pk_key, pe in param_errors.items():
        for k, v in pe.items():
            if k.endswith('_RE'):
                all_re.append(v)
    mean_param_re = np.mean(all_re)
    
    print(f"[EVAL]   Curve PSNR = {psnr:.2f} dB")
    print(f"[EVAL]   Curve CC   = {cc:.6f}")
    print(f"[EVAL]   Curve RMSE = {rmse:.6f}")
    print(f"[EVAL]   Residual std = {res_std:.6f}")
    print(f"[EVAL]   Mean param RE = {mean_param_re:.6f}")
    print("[EVAL] Done.")
    
    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------
    print("[VIS] Creating visualization ...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Spectral Peak Fitting – Inverse Problem", fontsize=16, fontweight='bold')
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # (a) Measured + Fitted + Individual peaks
    ax = axes[0, 0]
    ax.plot(x, measured_spectrum, 'k-', alpha=0.4, linewidth=0.5, label='Measured')
    ax.plot(x, fitted_spectrum, 'r-', linewidth=2, label='Fitted (total)')
    for i, y_fit in enumerate(individual_peaks_fit):
        ax.fill_between(x, fitted_baseline, fitted_baseline + y_fit, alpha=0.3, color=colors[i],
                         label=f'Peak {i} ({true_peaks[i]["type"]})')
    ax.plot(x, fitted_baseline, 'k--', linewidth=1, alpha=0.5, label='Baseline')
    ax.set_xlabel('Channel / Wavenumber')
    ax.set_ylabel('Intensity')
    ax.set_title('(a) Measured Spectrum + Fitted Decomposition')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # (b) Ground truth vs fitted
    ax = axes[0, 1]
    ax.plot(x, gt_signal, 'b-', linewidth=2, label='Ground Truth (clean)')
    ax.plot(x, fitted_spectrum, 'r--', linewidth=2, label='Fitted')
    ax.set_xlabel('Channel / Wavenumber')
    ax.set_ylabel('Intensity')
    ax.set_title(f'(b) Ground Truth vs Fitted  [PSNR={psnr:.1f} dB, CC={cc:.4f}]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (c) Residual
    ax = axes[1, 0]
    ax.plot(x, residuals, 'g-', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.fill_between(x, -2*res_std, 2*res_std, alpha=0.15, color='orange', label=f'±2σ (σ={res_std:.4f})')
    ax.set_xlabel('Channel / Wavenumber')
    ax.set_ylabel('Residual')
    ax.set_title('(c) Residual (Measured − Fitted)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (d) Parameter comparison table
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('(d) Parameter Comparison', fontsize=12, fontweight='bold')
    
    col_labels = ['Peak', 'Type', 'Param', 'True', 'Fitted', 'RE (%)']
    table_data = []
    for i, pk in enumerate(true_peaks):
        pe = param_errors[f'peak_{i}']
        table_data.append([f'P{i}', pk['type'][:4], 'Amp',
                           f"{pk['amplitude']:.2f}",
                           f"{pe['fitted_amplitude']:.2f}",
                           f"{pe['amplitude_RE']*100:.2f}"])
        table_data.append(['', '', 'Cen',
                           f"{pk['center']:.1f}",
                           f"{pe['fitted_center']:.1f}",
                           f"{pe['center_RE']*100:.2f}"])
        if pk['type'] in ('gaussian', 'voigt'):
            sig_fit = result.params[f'p{i}_sigma'].value
            table_data.append(['', '', 'σ',
                               f"{pk['sigma']:.2f}",
                               f"{sig_fit:.2f}",
                               f"{pe['sigma_RE']*100:.2f}"])
        if pk['type'] in ('lorentzian', 'voigt'):
            gam_fit = result.params[f'p{i}_gamma'].value
            table_data.append(['', '', 'γ',
                               f"{pk['gamma']:.2f}",
                               f"{gam_fit:.2f}",
                               f"{pe['gamma_RE']*100:.2f}"])
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved figure to {fig_path}")
    
    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    print("[SAVE] Saving outputs ...")
    
    gt_path = os.path.join(results_dir, "ground_truth.npy")
    recon_path = os.path.join(results_dir, "reconstruction.npy")
    np.save(gt_path, gt_signal)
    np.save(recon_path, fitted_spectrum)
    print(f"[SAVE]   ground_truth.npy  shape={gt_signal.shape}")
    print(f"[SAVE]   reconstruction.npy shape={fitted_spectrum.shape}")
    
    # Metrics JSON
    metrics = {
        "task": "spectrafit_peak",
        "inverse_problem": "Spectral peak fitting / spectral deconvolution",
        "method": "lmfit least-squares composite model fitting",
        "num_peaks": len(true_peaks),
        "peak_types": [pk["type"] for pk in true_peaks],
        "psnr_dB": round(float(psnr), 2),
        "correlation_coefficient": round(float(cc), 6),
        "rmse": round(float(rmse), 6),
        "residual_std": round(float(res_std), 6),
        "mean_parameter_relative_error": round(float(mean_param_re), 6),
        "snr_target_dB": snr_target,
        "num_data_points": len(x),
        "fit_converged": bool(result.success),
        "reduced_chi_square": round(float(result.redchi), 6),
        "num_function_evals": int(result.nfev),
        "per_peak_errors": {},
    }
    for i, pk in enumerate(true_peaks):
        pe = param_errors[f'peak_{i}']
        metrics["per_peak_errors"][f"peak_{i}_{pk['type']}_center{pk['center']}"] = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in pe.items()
        }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE]   metrics.json written")
    
    np.save(os.path.join(results_dir, "x_axis.npy"), x)
    np.save(os.path.join(results_dir, "measured_spectrum.npy"), measured_spectrum)
    
    print("[SAVE] All outputs saved.")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  PSNR  = {psnr:.2f} dB")
    print(f"  CC    = {cc:.6f}")
    print(f"  RMSE  = {rmse:.6f}")
    print(f"  Mean Parameter RE = {mean_param_re:.6f} ({mean_param_re*100:.2f}%)")
    print(f"  Fit converged: {result.success}")
    print(f"{'='*60}")
    
    return metrics
