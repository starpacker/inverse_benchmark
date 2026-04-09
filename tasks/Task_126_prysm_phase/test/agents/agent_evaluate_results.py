import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def evaluate_results(data: dict, inversion_result: dict, output_dir: str = 'results') -> dict:
    """
    Evaluate phase retrieval results and generate metrics/visualizations.
    
    Computes RMSE, PSNR, SSIM, correlation coefficient, Strehl ratios.
    Generates visualization plots and saves results.
    
    Returns metrics dictionary.
    """
    from skimage.metrics import structural_similarity as ssim
    
    os.makedirs(output_dir, exist_ok=True)
    
    pupil_mask = data['pupil_mask']
    pupil_bool = data['pupil_bool']
    true_phase_rad = data['true_phase_rad']
    retrieval_nms = data['retrieval_nms']
    zernike_specs = data['zernike_specs']
    psf_infocus_noisy = data['psf_infocus_noisy']
    psf_defocus_noisy = data['psf_defocus_noisy']
    
    coefs_opt = inversion_result['coefs_opt']
    retrieved_phase_rad = inversion_result['retrieved_phase_rad']
    
    # Center phases (remove piston)
    true_mean = np.sum(true_phase_rad * pupil_mask) / np.sum(pupil_mask)
    retr_mean = np.sum(retrieved_phase_rad * pupil_mask) / np.sum(pupil_mask)
    
    true_centered = (true_phase_rad - true_mean) * pupil_mask
    retrieved_centered = (retrieved_phase_rad - retr_mean) * pupil_mask
    
    error_map = (retrieved_centered - true_centered) * pupil_mask
    
    # Compute metrics
    phase_rmse_rad = np.sqrt(np.mean(error_map[pupil_bool]**2))
    phase_rmse_waves = phase_rmse_rad / (2 * np.pi)
    
    signal_range = np.ptp(true_centered[pupil_bool])
    phase_psnr = 20 * np.log10(signal_range / phase_rmse_rad) if phase_rmse_rad > 0 else float('inf')
    
    cc = np.corrcoef(true_centered[pupil_bool], retrieved_centered[pupil_bool])[0, 1]
    
    true_rms = np.std(true_centered[pupil_bool])
    retr_rms = np.std(retrieved_centered[pupil_bool])
    strehl_true = np.exp(-true_rms**2)
    strehl_retrieved = np.exp(-retr_rms**2)
    
    # SSIM
    vmin = min(true_centered[pupil_bool].min(), retrieved_centered[pupil_bool].min())
    vmax = max(true_centered[pupil_bool].max(), retrieved_centered[pupil_bool].max())
    drange = vmax - vmin if (vmax - vmin) > 1e-10 else 1.0
    true_norm = (true_centered - vmin) / drange * pupil_mask
    retr_norm = (retrieved_centered - vmin) / drange * pupil_mask
    ssim_val = ssim(true_norm, retr_norm, data_range=1.0)
    
    # Print coefficient comparison
    print(f"\n{'Mode':<12} {'True (waves)':<14} {'Retrieved (waves)':<18} {'Error (waves)':<14}")
    print("-" * 58)
    
    truth_dict = {(n, m): c for n, m, c in zernike_specs}
    for i, (n, m) in enumerate(retrieval_nms):
        true_c = truth_dict.get((n, m), 0.0)
        retr_c = coefs_opt[i]
        err_c = retr_c - true_c
        name = f"Z({n},{m:+d})"
        if abs(true_c) > 0 or abs(retr_c) > 0.01:
            print(f"  {name:<10} {true_c:>12.4f}   {retr_c:>14.4f}     {err_c:>12.4f}")
    
    print(f"\n{'='*50}")
    print(f"Phase Retrieval Results:")
    print(f"{'='*50}")
    print(f"Phase RMSE:      {phase_rmse_rad:.4f} rad ({phase_rmse_waves:.4f} waves)")
    print(f"Phase PSNR:      {phase_psnr:.2f} dB")
    print(f"SSIM:            {ssim_val:.4f}")
    print(f"Correlation:     {cc:.6f}")
    print(f"Strehl (true):   {strehl_true:.6f}")
    print(f"Strehl (retr):   {strehl_retrieved:.6f}")
    print(f"{'='*50}")
    
    # Build metrics dictionary
    metrics = {
        'task': 'prysm_phase',
        'task_number': 126,
        'method': 'Parametric phase diversity optimization (L-BFGS-B)',
        'phase_rmse_rad': round(float(phase_rmse_rad), 4),
        'phase_rmse_waves': round(float(phase_rmse_waves), 4),
        'phase_psnr_dB': round(float(phase_psnr), 2),
        'ssim': round(float(ssim_val), 4),
        'correlation_coefficient': round(float(cc), 6),
        'strehl_ratio_true': round(float(strehl_true), 6),
        'strehl_ratio_retrieved': round(float(strehl_retrieved), 6),
        'grid_size': data['npix'],
        'n_zernike_retrieval_modes': data['n_modes'],
        'zernike_modes_truth': [{'n': n, 'm': m, 'coeff_waves': float(c)} for n, m, c in zernike_specs],
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved results/metrics.json")
    
    # Save arrays
    np.save(os.path.join(output_dir, 'ground_truth.npy'), true_centered)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), retrieved_centered)
    print("Saved results/ground_truth.npy, results/reconstruction.npy")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    phase_vmin = np.min(true_centered[pupil_bool])
    phase_vmax = np.max(true_centered[pupil_bool])
    
    # (0,0) Ground truth phase
    ax = axes[0, 0]
    disp = true_centered.copy()
    disp[~pupil_bool] = np.nan
    im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    ax.set_title('Ground Truth Phase (rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
    ax.axis('off')
    
    # (0,1) Measured in-focus PSF
    ax = axes[0, 1]
    psf_d = psf_infocus_noisy.copy()
    psf_d[psf_d <= 0] = 1e-10
    im = ax.imshow(np.log10(psf_d), cmap='inferno')
    ax.set_title('Measured PSF (in-focus, log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
    ax.axis('off')
    
    # (0,2) Measured defocused PSF
    ax = axes[0, 2]
    psf_d2 = psf_defocus_noisy.copy()
    psf_d2[psf_d2 <= 0] = 1e-10
    im = ax.imshow(np.log10(psf_d2), cmap='inferno')
    ax.set_title('Measured PSF (defocused, log10)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10(I)')
    ax.axis('off')
    
    # (1,0) Retrieved phase
    ax = axes[1, 0]
    disp = retrieved_centered.copy()
    disp[~pupil_bool] = np.nan
    im = ax.imshow(disp, cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    ax.set_title('Retrieved Phase (rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Phase (rad)')
    ax.axis('off')
    
    # (1,1) Phase error
    ax = axes[1, 1]
    disp = error_map.copy()
    disp[~pupil_bool] = np.nan
    err_lim = max(abs(np.nanmin(disp)), abs(np.nanmax(disp)), 0.01)
    im = ax.imshow(disp, cmap='RdBu_r', vmin=-err_lim, vmax=err_lim)
    ax.set_title(f'Phase Error (RMSE={phase_rmse_rad:.4f} rad)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Error (rad)')
    ax.axis('off')
    
    # (1,2) Coefficient comparison
    ax = axes[1, 2]
    mode_labels = []
    true_vals = []
    retr_vals = []
    for i, (n, m) in enumerate(retrieval_nms):
        true_c = truth_dict.get((n, m), 0.0)
        if abs(true_c) > 0 or abs(coefs_opt[i]) > 0.005:
            mode_labels.append(f"Z({n},{m:+d})")
            true_vals.append(true_c)
            retr_vals.append(coefs_opt[i])
    
    x_pos = np.arange(len(mode_labels))
    width = 0.35
    ax.bar(x_pos - width/2, true_vals, width, label='True', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, retr_vals, width, label='Retrieved', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient (waves)', fontsize=11)
    ax.set_title('Zernike Coefficients', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Phase Retrieval: Parametric Diversity Optimization (prysm)\n'
                 f'PSNR={phase_psnr:.2f} dB | SSIM={ssim_val:.4f} | CC={cc:.4f} | '
                 f'RMSE={phase_rmse_waves:.4f} waves',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved results/reconstruction_result.png")
    
    return metrics
