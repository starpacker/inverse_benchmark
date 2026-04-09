import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def evaluate_results(data, result, output_dir='results'):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data containing ground truth.
    result : dict
        Dictionary from run_inversion containing reconstruction results.
    output_dir : str
        Directory to save outputs.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    wavenumber = data['wavenumber']
    pure_components = data['pure_components']
    true_weights = data['true_weights']
    clean_spectra = data['clean_spectra']
    baselines = data['baselines']
    observed_spectra = data['observed_spectra']
    n_components = data['n_components']
    
    corrected_spectra = result['corrected_spectra']
    fitted_baselines = result['fitted_baselines']
    H_matched = result['recovered_components']
    W_matched = result['recovered_weights']
    col_ind = result['permutation']
    
    n_mixtures = observed_spectra.shape[0]
    
    # ============================================================
    # EVALUATION METRICS
    # ============================================================
    
    metrics = {}
    
    # --- (a) Baseline-corrected PSNR ---
    # Compare corrected spectra vs clean spectra (ground truth without baseline/noise)
    mse_baseline = np.mean((corrected_spectra - clean_spectra) ** 2)
    max_val = clean_spectra.max()
    psnr_baseline = 10 * np.log10(max_val ** 2 / mse_baseline)
    metrics['baseline_corrected_PSNR_dB'] = float(round(psnr_baseline, 2))
    
    # Correlation between corrected and clean
    cc_baseline_list = []
    for i in range(n_mixtures):
        cc = np.corrcoef(corrected_spectra[i], clean_spectra[i])[0, 1]
        cc_baseline_list.append(cc)
    metrics['baseline_corrected_mean_CC'] = float(round(np.mean(cc_baseline_list), 4))
    
    # --- (b) Component spectrum correlation ---
    comp_cc_list = []
    for i in range(n_components):
        cc = np.corrcoef(pure_components[i], H_matched[i])[0, 1]
        comp_cc_list.append(cc)
        metrics[f'component_{i+1}_CC'] = float(round(cc, 4))
    metrics['mean_component_CC'] = float(round(np.mean(comp_cc_list), 4))
    
    # --- (c) Mixing proportion relative error ---
    re_per_sample = np.abs(W_matched - true_weights) / (np.abs(true_weights) + 1e-10)
    mean_re = np.mean(re_per_sample)
    metrics['mixing_proportion_mean_RE'] = float(round(mean_re, 4))
    
    # Mixing proportion correlation (per component)
    for i in range(n_components):
        cc_w = np.corrcoef(true_weights[:, i], W_matched[:, i])[0, 1]
        metrics[f'weight_component_{i+1}_CC'] = float(round(cc_w, 4))
    
    # Overall PSNR (as a summary metric)
    metrics['PSNR'] = metrics['baseline_corrected_PSNR_dB']
    
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # ============================================================
    # SAVE GROUND TRUTH AND RECONSTRUCTION
    # ============================================================
    
    # Ground truth: dict with all true data
    gt_data = {
        'wavenumber': wavenumber,
        'pure_components': pure_components,
        'true_weights': true_weights,
        'clean_spectra': clean_spectra,
        'baselines': baselines,
        'observed_spectra': observed_spectra,
    }
    np.save(os.path.join(output_dir, 'ground_truth.npy'), gt_data, allow_pickle=True)
    
    # Reconstruction: dict with all recovered data
    recon_data = {
        'corrected_spectra': corrected_spectra,
        'recovered_components': H_matched,
        'recovered_weights': W_matched,
        'fitted_baselines': fitted_baselines,
        'permutation': col_ind,
    }
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_data, allow_pickle=True)
    
    # ============================================================
    # VISUALIZATION – 6 subplots
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # --- (a) Example mixed spectrum: observed vs corrected vs clean ---
    ax = axes[0, 0]
    idx = 0  # first mixture
    ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.6, label='Observed (with baseline+noise)')
    ax.plot(wavenumber, corrected_spectra[idx], 'r-', linewidth=1.5, label='Baseline corrected')
    ax.plot(wavenumber, clean_spectra[idx], 'k--', linewidth=1.5, label='Ground truth (clean)')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('(a) Baseline Correction Example')
    ax.legend(fontsize=8)
    
    # --- (b) Baseline correction detail ---
    ax = axes[0, 1]
    ax.plot(wavenumber, observed_spectra[idx], 'b-', alpha=0.5, label='Observed')
    ax.plot(wavenumber, fitted_baselines[idx], 'g-', linewidth=2, label='Fitted baseline')
    ax.plot(wavenumber, baselines[idx], 'm--', linewidth=2, label='True baseline')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('(b) True vs Fitted Baseline')
    ax.legend(fontsize=8)
    
    # --- (c) True vs recovered component spectra ---
    ax = axes[0, 2]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = ['Mineral A', 'Mineral B', 'Mineral C']
    for i in range(n_components):
        ax.plot(wavenumber, pure_components[i], '-', color=colors[i],
                linewidth=2, label=f'True {labels[i]}')
        ax.plot(wavenumber, H_matched[i], '--', color=colors[i],
                linewidth=1.5, alpha=0.8, label=f'Recovered (CC={comp_cc_list[i]:.3f})')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('(c) True vs Recovered Components')
    ax.legend(fontsize=7, ncol=2)
    
    # --- (d) Mixing proportions: true vs recovered (scatter) ---
    ax = axes[1, 0]
    for i in range(n_components):
        ax.scatter(true_weights[:, i], W_matched[:, i], color=colors[i],
                   s=40, label=f'{labels[i]}', alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Ideal')
    ax.set_xlabel('True Mixing Proportion')
    ax.set_ylabel('Recovered Mixing Proportion')
    ax.set_title('(d) Mixing Proportions: True vs Recovered')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # --- (e) Residual after baseline correction ---
    ax = axes[1, 1]
    residuals = corrected_spectra - clean_spectra
    for i in range(min(5, n_mixtures)):
        ax.plot(wavenumber, residuals[i], alpha=0.5, linewidth=0.8, label=f'Mix {i+1}')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Residual')
    ax.set_title(f'(e) Baseline Correction Residuals (PSNR={psnr_baseline:.1f} dB)')
    ax.legend(fontsize=7)
    
    # --- (f) Per-mixture weight recovery bar chart ---
    ax = axes[1, 2]
    mix_indices = np.arange(n_mixtures)
    bar_width = 0.12
    for i in range(n_components):
        ax.bar(mix_indices - bar_width + i * bar_width, true_weights[:, i],
               bar_width, color=colors[i], alpha=0.5, label=f'True {labels[i]}')
        ax.bar(mix_indices + i * bar_width, W_matched[:, i],
               bar_width, color=colors[i], edgecolor='k', linewidth=0.5,
               label=f'Rec {labels[i]}')
    ax.set_xlabel('Mixture Index')
    ax.set_ylabel('Weight')
    ax.set_title('(f) Weight Recovery per Mixture')
    # Simplify legend
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[:6], lbls[:6], fontsize=6, ncol=2)
    ax.set_xticks(mix_indices)
    
    plt.suptitle('Raman Spectral Unmixing & Baseline Correction (rampy + NMF)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}/reconstruction_result.png")
    print(f"Metrics saved to {output_dir}/metrics.json")
    print(f"Ground truth saved to {output_dir}/ground_truth.npy")
    print(f"Reconstruction saved to {output_dir}/reconstruction.npy")
    
    return metrics
