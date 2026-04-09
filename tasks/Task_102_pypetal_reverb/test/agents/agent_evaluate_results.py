import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(data, inversion_result, results_dir, assets_dir):
    """
    Compute metrics, generate visualizations, and save outputs.
    
    Parameters:
    -----------
    data : dict
        Preprocessed data dictionary
    inversion_result : dict
        Results from run_inversion containing recovered transfer function
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
    
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, CC, RMSE, and peak_lag_error
    """
    psi_gt = data['psi_gt']
    psi_rec = inversion_result['psi_rec']
    tau = data['tau']
    dt = data['dt']
    tau_peak = data['tau_peak']
    t = data['t']
    continuum = data['continuum']
    line_clean = data['line_clean']
    line_obs = data['line_obs']
    ccf_lags = inversion_result['ccf_lags']
    ccf = inversion_result['ccf']
    
    # Compute metrics
    max_lag = 80.0
    mask = tau <= max_lag
    gt = psi_gt[mask]
    rec = psi_rec[mask]
    
    # Normalise both to peak = 1 for scale-invariant comparison
    gt_peak = gt.max()
    rec_peak = rec.max()
    gt_norm = gt / gt_peak if gt_peak > 0 else gt
    rec_norm = rec / rec_peak if rec_peak > 0 else rec
    
    mse = np.mean((gt_norm - rec_norm)**2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(gt_norm, rec_norm)[0, 1])
    rmse = float(np.sqrt(mse))
    
    # Peak lag recovery
    peak_gt = tau[mask][np.argmax(gt)]
    peak_rec = tau[mask][np.argmax(rec)]
    peak_error = abs(peak_rec - peak_gt)
    
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "peak_lag_error": float(peak_error),
    }
    
    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    Peak lag error = {peak_error:.1f} days")
    
    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), psi_gt)
        np.save(os.path.join(d, "recon_output.npy"), psi_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ccf_mask = ccf_lags <= max_lag
    
    # Panel 1: Light curves
    ax = axes[0, 0]
    ax.plot(t, continuum, 'b-', lw=0.8, label='Continuum C(t)')
    ax.plot(t, line_obs, 'r-', lw=0.8, alpha=0.6, label='Line L(t) [observed]')
    ax.plot(t, line_clean, 'g--', lw=0.8, label='Line L(t) [clean]')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("AGN Light Curves")
    ax.legend(fontsize=8)
    
    # Panel 2: Transfer function
    ax = axes[0, 1]
    ax.plot(tau[mask], psi_gt[mask], 'b-', lw=2, label='GT  Ψ(τ)')
    ax.plot(tau[mask], rec_norm, 'r--', lw=2, label='Recovered Ψ(τ)')
    ax.axvline(tau_peak, color='gray', ls=':', lw=1, label=f'True peak = {tau_peak} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Ψ(τ)")
    ax.set_title(f"Transfer Function | PSNR={metrics['PSNR']:.1f} dB, CC={metrics['CC']:.4f}")
    ax.legend(fontsize=8)
    
    # Panel 3: CCF
    ax = axes[1, 0]
    ax.plot(ccf_lags[ccf_mask], ccf[ccf_mask], 'k-', lw=1.5)
    ax.axvline(tau_peak, color='r', ls='--', lw=1, label=f'True lag = {tau_peak} d')
    peak_ccf_lag = ccf_lags[ccf_mask][np.argmax(ccf[ccf_mask])]
    ax.axvline(peak_ccf_lag, color='b', ls=':', lw=1, label=f'CCF peak = {peak_ccf_lag:.1f} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("CCF")
    ax.set_title("Cross-Correlation Function")
    ax.legend(fontsize=8)
    
    # Panel 4: Residual of transfer function
    ax = axes[1, 1]
    residual = gt_norm - rec_norm
    ax.plot(tau[mask], residual, 'k-', lw=1)
    ax.axhline(0, color='r', ls='--', lw=0.5)
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Ψ Residual | Peak error = {metrics['peak_lag_error']:.1f} days")
    ax.fill_between(tau[mask], residual, alpha=0.3, color='gray')
    
    plt.tight_layout()
    for path in [os.path.join(results_dir, "vis_result.png"),
                 os.path.join(assets_dir, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return metrics
