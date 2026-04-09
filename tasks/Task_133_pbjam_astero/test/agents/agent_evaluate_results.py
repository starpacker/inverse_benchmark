import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(gt_freq_array, recon_result, clean_spectrum, metadata, results_dir):
    """
    Compute evaluation metrics and generate visualizations for peakbagging.
    
    Args:
        gt_freq_array: ndarray, ground truth mode frequencies
        recon_result: dict from run_inversion
        clean_spectrum: ndarray, the clean power spectrum
        metadata: dict containing auxiliary data
        results_dir: str, path to save results
        
    Returns:
        metrics: dict containing all evaluation metrics
    """
    matched_gt = recon_result['matched_gt']
    matched_fitted = recon_result['matched_fitted']
    recon_spectrum = recon_result['recon_spectrum']
    fitted_freqs = recon_result['fitted_freqs']
    freqs = metadata['freqs']
    noisy_spectrum = metadata.get('noisy_spectrum', None)
    delta_nu = metadata['delta_nu']
    freq_min = metadata['freq_min']
    freq_max = metadata['freq_max']
    gt_modes = metadata['gt_modes']
    smoothed = metadata['smoothed_spectrum']
    
    metrics = {}
    
    if len(matched_gt) > 0:
        # Frequency relative errors
        freq_errors = np.abs(matched_fitted - matched_gt)
        rel_errors = freq_errors / matched_gt * 100
        
        metrics['mean_freq_error_uHz'] = float(np.mean(freq_errors))
        metrics['median_freq_error_uHz'] = float(np.median(freq_errors))
        metrics['max_freq_error_uHz'] = float(np.max(freq_errors))
        metrics['mean_relative_error_pct'] = float(np.mean(rel_errors))
        metrics['median_relative_error_pct'] = float(np.median(rel_errors))
        
        if len(matched_gt) > 1:
            cc = float(np.corrcoef(matched_gt, matched_fitted)[0, 1])
        else:
            cc = 1.0
        metrics['frequency_CC'] = cc
        
        ss_res = np.sum((matched_gt - matched_fitted) ** 2)
        ss_tot = np.sum((matched_gt - np.mean(matched_gt)) ** 2)
        metrics['frequency_R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        metrics['n_gt_modes'] = len(gt_freq_array)
        metrics['n_detected'] = len(fitted_freqs)
        metrics['n_matched'] = len(matched_gt)
        metrics['detection_rate'] = float(len(matched_gt) / len(gt_freq_array))
    
    if clean_spectrum is not None and recon_spectrum is not None:
        data_range = clean_spectrum.max() - clean_spectrum.min()
        mse = np.mean((clean_spectrum - recon_spectrum) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(data_range ** 2 / mse)
        else:
            psnr = float('inf')
        metrics['spectrum_PSNR'] = float(psnr)
        
        cc_spec = float(np.corrcoef(clean_spectrum, recon_spectrum)[0, 1])
        metrics['spectrum_CC'] = cc_spec
    
    if len(matched_gt) >= 4:
        sorted_fitted = np.sort(matched_fitted)
        diffs = np.diff(sorted_fitted)
        dnu_candidates = diffs[(diffs > delta_nu * 0.7) & (diffs < delta_nu * 1.3)]
        if len(dnu_candidates) > 0:
            fitted_dnu = np.median(dnu_candidates)
            metrics['fitted_delta_nu'] = float(fitted_dnu)
            metrics['delta_nu_error_pct'] = float(abs(fitted_dnu - delta_nu) / delta_nu * 100)
    
    # Print metrics
    print(f"\n[EVAL] === Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Ground truth clean spectrum with mode locations
    ax1 = axes[0, 0]
    ax1.plot(freqs, clean_spectrum, 'b-', alpha=0.8, linewidth=0.5, label='Clean spectrum')
    for (n, l), freq in sorted(gt_modes.items()):
        colors = {0: 'red', 1: 'green', 2: 'blue'}
        ax1.axvline(freq, color=colors.get(l, 'gray'), alpha=0.4, linewidth=0.5)
    ax1.set_xlabel('Frequency (μHz)')
    ax1.set_ylabel('Power (ppm²/μHz)')
    ax1.set_title('(a) Ground Truth Power Spectrum + Mode Frequencies')
    ax1.set_xlim(freq_min, freq_max)
    ax1.legend(fontsize=8)
    
    # Panel 2: Noisy observed spectrum
    ax2 = axes[0, 1]
    if noisy_spectrum is not None:
        ax2.plot(freqs, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.3, label='Noisy')
    ax2.plot(freqs, smoothed, 'b-', alpha=0.8, linewidth=0.8, label='Smoothed')
    ax2.plot(freqs, recon_result['bg_estimate'], 'r--', alpha=0.7, linewidth=1, label='Background')
    for ff in fitted_freqs:
        ax2.axvline(ff, color='orange', alpha=0.4, linewidth=0.5)
    ax2.set_xlabel('Frequency (μHz)')
    ax2.set_ylabel('Power (ppm²/μHz)')
    ax2.set_title(f'(b) Observed Spectrum + {len(fitted_freqs)} Detected Peaks')
    ax2.set_xlim(freq_min, freq_max)
    ax2.legend(fontsize=8)
    
    # Panel 3: Reconstructed spectrum vs clean
    ax3 = axes[1, 0]
    ax3.plot(freqs, clean_spectrum, 'b-', alpha=0.6, linewidth=0.8, label='Clean GT')
    ax3.plot(freqs, recon_spectrum, 'r-', alpha=0.6, linewidth=0.8, label='Reconstructed')
    ax3.set_xlabel('Frequency (μHz)')
    ax3.set_ylabel('Power (ppm²/μHz)')
    psnr_val = metrics.get('spectrum_PSNR', 0)
    cc_val = metrics.get('spectrum_CC', 0)
    ax3.set_title(f'(c) Spectrum Reconstruction — PSNR={psnr_val:.2f} dB, CC={cc_val:.4f}')
    ax3.set_xlim(freq_min, freq_max)
    ax3.legend(fontsize=8)
    
    # Panel 4: Frequency comparison (GT vs Fitted)
    ax4 = axes[1, 1]
    if len(matched_gt) > 0:
        ax4.scatter(matched_gt, matched_fitted, c='blue', s=50, zorder=5, label='Matched modes')
        fmin_plot, fmax_plot = matched_gt.min() - 20, matched_gt.max() + 20
        ax4.plot([fmin_plot, fmax_plot], [fmin_plot, fmax_plot], 'k--', alpha=0.5, label='Perfect match')
        
        errors = matched_fitted - matched_gt
        for i in range(len(matched_gt)):
            ax4.annotate(f'{errors[i]:.2f}', (matched_gt[i], matched_fitted[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
        
        freq_cc = metrics.get('frequency_CC', 0)
        mean_re = metrics.get('mean_relative_error_pct', 0)
        det_rate = metrics.get('detection_rate', 0)
        ax4.set_title(f'(d) Freq Match — CC={freq_cc:.6f}, RE={mean_re:.4f}%, Det={det_rate:.1%}')
    ax4.set_xlabel('Ground Truth Frequency (μHz)')
    ax4.set_ylabel('Fitted Frequency (μHz)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), matched_fitted)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_freq_array)
    if noisy_spectrum is not None:
        np.save(os.path.join(results_dir, "noisy_spectrum.npy"), noisy_spectrum)
    np.save(os.path.join(results_dir, "clean_spectrum.npy"), clean_spectrum)
    np.save(os.path.join(results_dir, "recon_spectrum.npy"), recon_spectrum)
    
    return metrics
