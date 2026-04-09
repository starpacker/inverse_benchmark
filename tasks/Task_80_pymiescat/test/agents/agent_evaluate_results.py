import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(ground_truth, recon_params, recon_spectra, observations, results_dir):
    """
    Compute comprehensive evaluation metrics and generate visualizations.
    
    Args:
        ground_truth: dict with true parameters and clean spectra
        recon_params: dict with recovered n_real, k_imag, diameter
        recon_spectra: dict with reconstructed qsca, qabs, qext, g arrays
        observations: dict with wavelengths and noisy observations
        results_dir: directory path to save results
    
    Returns:
        metrics: dict with all evaluation metrics
    """
    # Parameter recovery errors
    n_error = abs(recon_params['n_real'] - ground_truth['n_real'])
    k_error = abs(recon_params['k_imag'] - ground_truth['k_imag'])
    d_error = abs(recon_params['diameter'] - ground_truth['diameter'])
    
    n_re = n_error / ground_truth['n_real']
    k_re = k_error / ground_truth['k_imag']
    d_re = d_error / ground_truth['diameter']
    
    # Spectral reconstruction quality
    gt_qsca = ground_truth['qsca_clean']
    gt_qabs = ground_truth['qabs_clean']
    gt_qext = ground_truth['qext_clean']
    
    rec_qsca = recon_spectra['qsca']
    rec_qabs = recon_spectra['qabs']
    rec_qext = recon_spectra['qext']
    
    # RMSE for spectra
    rmse_qsca = np.sqrt(np.mean((gt_qsca - rec_qsca)**2))
    rmse_qabs = np.sqrt(np.mean((gt_qabs - rec_qabs)**2))
    rmse_qext = np.sqrt(np.mean((gt_qext - rec_qext)**2))
    
    # Correlation coefficients
    cc_qsca = np.corrcoef(gt_qsca, rec_qsca)[0, 1]
    cc_qabs = np.corrcoef(gt_qabs, rec_qabs)[0, 1]
    cc_qext = np.corrcoef(gt_qext, rec_qext)[0, 1]
    
    # PSNR for Qext spectrum
    data_range = gt_qext.max() - gt_qext.min()
    mse_qext = np.mean((gt_qext - rec_qext)**2)
    psnr = 10 * np.log10(data_range**2 / mse_qext) if mse_qext > 0 else float('inf')
    
    # Overall relative error
    gt_all = np.concatenate([gt_qsca, gt_qabs])
    rec_all = np.concatenate([rec_qsca, rec_qabs])
    overall_re = np.sqrt(np.mean((gt_all - rec_all)**2)) / np.sqrt(np.mean(gt_all**2))
    
    metrics = {
        'n_real_gt': ground_truth['n_real'],
        'n_real_recon': recon_params['n_real'],
        'n_real_error': float(n_error),
        'n_real_relative_error': float(n_re),
        'k_imag_gt': ground_truth['k_imag'],
        'k_imag_recon': recon_params['k_imag'],
        'k_imag_error': float(k_error),
        'k_imag_relative_error': float(k_re),
        'diameter_gt': ground_truth['diameter'],
        'diameter_recon': recon_params['diameter'],
        'diameter_error': float(d_error),
        'diameter_relative_error': float(d_re),
        'rmse_qsca': float(rmse_qsca),
        'rmse_qabs': float(rmse_qabs),
        'rmse_qext': float(rmse_qext),
        'cc_qsca': float(cc_qsca),
        'cc_qabs': float(cc_qabs),
        'cc_qext': float(cc_qext),
        'psnr': float(psnr),
        'rmse': float(rmse_qext),
        'overall_relative_error': float(overall_re),
    }
    
    # Print metrics
    print(f"[EVAL] n error = {metrics['n_real_error']:.6f} "
          f"(RE: {metrics['n_real_relative_error']:.6f})")
    print(f"[EVAL] k error = {metrics['k_imag_error']:.6f} "
          f"(RE: {metrics['k_imag_relative_error']:.6f})")
    print(f"[EVAL] d error = {metrics['diameter_error']:.2f} nm "
          f"(RE: {metrics['diameter_relative_error']:.6f})")
    print(f"[EVAL] CC_Qsca = {metrics['cc_qsca']:.6f}")
    print(f"[EVAL] CC_Qabs = {metrics['cc_qabs']:.6f}")
    print(f"[EVAL] CC_Qext = {metrics['cc_qext']:.6f}")
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] Overall RE = {metrics['overall_relative_error']:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    gt_spectra = np.stack([ground_truth['qsca_clean'], 
                           ground_truth['qabs_clean'],
                           ground_truth['qext_clean']], axis=0)
    rec_spectra_arr = np.stack([recon_spectra['qsca'],
                                recon_spectra['qabs'],
                                recon_spectra['qext']], axis=0)
    input_data = np.stack([observations['qsca'],
                           observations['qabs'],
                           observations['qext']], axis=0)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_spectra)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_spectra_arr)
    np.save(os.path.join(results_dir, "input.npy"), input_data)
    print(f"[SAVE] GT spectra shape: {gt_spectra.shape} → ground_truth.npy")
    print(f"[SAVE] Recon spectra shape: {rec_spectra_arr.shape} → reconstruction.npy")
    print(f"[SAVE] Input spectra shape: {input_data.shape} → input.npy")
    
    # Generate visualization
    wl = observations['wavelengths']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Qsca spectrum comparison
    axes[0, 0].plot(wl, ground_truth['qsca_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 0].plot(wl, observations['qsca'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 0].plot(wl, recon_spectra['qsca'], 'r--', lw=2, label='Reconstructed')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Qsca')
    axes[0, 0].set_title(f'Scattering Efficiency (CC={metrics["cc_qsca"]:.6f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Qabs spectrum comparison
    axes[0, 1].plot(wl, ground_truth['qabs_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 1].plot(wl, observations['qabs'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 1].plot(wl, recon_spectra['qabs'], 'r--', lw=2, label='Reconstructed')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Qabs')
    axes[0, 1].set_title(f'Absorption Efficiency (CC={metrics["cc_qabs"]:.6f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Qext spectrum comparison
    axes[0, 2].plot(wl, ground_truth['qext_clean'], 'b-', lw=2, label='GT (clean)')
    axes[0, 2].plot(wl, observations['qext'], 'k.', ms=4, alpha=0.5, label='Observed (noisy)')
    axes[0, 2].plot(wl, recon_spectra['qext'], 'r--', lw=2, label='Reconstructed')
    axes[0, 2].set_xlabel('Wavelength (nm)')
    axes[0, 2].set_ylabel('Qext')
    axes[0, 2].set_title(f'Extinction Efficiency (CC={metrics["cc_qext"]:.6f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) Residuals
    res_sca = ground_truth['qsca_clean'] - recon_spectra['qsca']
    res_abs = ground_truth['qabs_clean'] - recon_spectra['qabs']
    axes[1, 0].plot(wl, res_sca, 'b-', lw=1.5, label='Qsca residual')
    axes[1, 0].plot(wl, res_abs, 'r-', lw=1.5, label='Qabs residual')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Spectral Residuals (GT - Recon)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (e) Parameter comparison bar chart
    params_names = ['n (real)', 'k (imag)', 'd (nm)']
    gt_vals = [ground_truth['n_real'], ground_truth['k_imag'], ground_truth['diameter']]
    rec_vals = [recon_params['n_real'], recon_params['k_imag'], recon_params['diameter']]
    
    # Normalize for display
    gt_norm = np.array(gt_vals) / np.array(gt_vals)
    rec_norm = np.array(rec_vals) / np.array(gt_vals)
    
    x_pos = np.arange(len(params_names))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, gt_norm, width, label='Ground Truth', color='steelblue')
    axes[1, 1].bar(x_pos + width/2, rec_norm, width, label='Reconstructed', color='coral')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(params_names)
    axes[1, 1].set_ylabel('Normalized Value (GT=1.0)')
    axes[1, 1].set_title('Parameter Recovery')
    axes[1, 1].legend()
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add error annotations
    for i, (name, gt, rec) in enumerate(zip(params_names, gt_vals, rec_vals)):
        re = abs(rec - gt) / gt * 100
        axes[1, 1].annotate(f'{re:.2f}%', xy=(i, max(gt_norm[i], rec_norm[i]) + 0.02),
                            ha='center', fontsize=9, color='red')
    
    # (f) Asymmetry parameter
    axes[1, 2].plot(wl, ground_truth['g_clean'], 'b-', lw=2, label='GT g(λ)')
    axes[1, 2].plot(wl, recon_spectra['g'], 'r--', lw=2, label='Recon g(λ)')
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('g (asymmetry parameter)')
    axes[1, 2].set_title('Asymmetry Parameter')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"PyMieScatt — Mie Scattering Refractive Index Inversion\n"
        f"GT: n={ground_truth['n_real']}, k={ground_truth['k_imag']}, d={ground_truth['diameter']} nm | "
        f"Recon: n={recon_params['n_real']:.4f}, k={recon_params['k_imag']:.6f}, d={recon_params['diameter']:.2f} nm\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC_ext={metrics['cc_qext']:.6f} | RE={metrics['overall_relative_error']:.6f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics
