import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings('ignore')

def forward_operator(kh, beta, zt, dz, C):
    """
    Forward: Compute the theoretical radial power spectrum from
    Curie depth parameters using Bouligand et al. (2009) model.
    
    The Bouligand model computes ln(Phi) where Phi is the radial power spectrum.
    
    Parameters:
        kh: Wavenumber array (rad/km)
        beta: Fractal parameter
        zt: Top of magnetic layer (km)
        dz: Thickness of magnetic layer (km)
        C: Field constant
    
    Returns:
        log_phi: Natural log of the radial power spectrum
    """
    from pycurious import bouligand2009
    return bouligand2009(kh, beta, zt, dz, C)

def evaluate_results(fitted_results, grid_obj, params, centroids, window_size, results_dir):
    """
    Evaluate Curie depth inversion quality and generate visualizations.
    
    Parameters:
        fitted_results: List of fitted parameter dictionaries
        grid_obj: CurieOptimise object
        params: Dictionary with true parameters
        centroids: List of centroid coordinates
        window_size: Window size used for spectral analysis
        results_dir: Directory to save results
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    true_beta = params['true_beta']
    true_zt = params['true_zt']
    true_dz = params['true_dz']
    true_c = params['true_c']
    true_curie = params['true_curie_depth']
    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']
    
    fitted_curie = []
    fitted_betas = []
    fitted_zts = []
    fitted_dzs = []
    
    for r in fitted_results:
        if not np.isnan(r['curie_depth']):
            fitted_curie.append(r['curie_depth'])
            fitted_betas.append(r['beta'])
            fitted_zts.append(r['zt'])
            fitted_dzs.append(r['dz'])
    
    fit_cd = np.array(fitted_curie)
    
    # Mean fitted values
    mean_cd = np.mean(fit_cd)
    std_cd = np.std(fit_cd)
    mean_beta = np.mean(fitted_betas)
    mean_zt = np.mean(fitted_zts)
    mean_dz = np.mean(fitted_dzs)
    
    # Curie depth error
    rmse_cd = np.sqrt(np.mean((fit_cd - true_curie)**2))
    rel_err_cd = np.abs(mean_cd - true_curie) / true_curie
    
    # Parameter relative errors (mean fitted vs true)
    rel_err_beta = np.abs(mean_beta - true_beta) / true_beta
    rel_err_zt = np.abs(mean_zt - true_zt) / true_zt
    rel_err_dz = np.abs(mean_dz - true_dz) / true_dz
    
    # Spectral fit quality at first centroid
    xc, yc = centroids[0]
    subgrid = grid_obj.subgrid(window_size, xc, yc)
    k, Phi_obs, sigma_Phi = grid_obj.radial_spectrum(subgrid, taper=np.hanning)
    
    # Forward model at fitted params
    fit_r = fitted_results[0]
    Phi_fit = forward_operator(k, fit_r['beta'], fit_r['zt'], fit_r['dz'], fit_r['C'])
    
    # True model spectrum
    Phi_true = forward_operator(k, true_beta, true_zt, true_dz, true_c)
    
    # Spectral CC
    valid = np.isfinite(Phi_obs) & np.isfinite(Phi_fit)
    cc_spec = np.corrcoef(Phi_obs[valid], Phi_fit[valid])[0, 1] if np.sum(valid) > 2 else float('nan')
    
    # PSNR of spectrum (fitted vs observed)
    mse_spec = np.mean((Phi_obs[valid] - Phi_fit[valid])**2)
    range_spec = Phi_obs[valid].max() - Phi_obs[valid].min()
    psnr_spec = 10 * np.log10(range_spec**2 / mse_spec) if mse_spec > 0 else float('inf')
    
    metrics = {
        'psnr_spectrum': float(psnr_spec),
        'cc_spectrum': float(cc_spec),
        'rmse_curie_depth_km': float(rmse_cd),
        'mean_curie_depth_km': float(mean_cd),
        'std_curie_depth_km': float(std_cd),
        'true_curie_depth_km': float(true_curie),
        'rel_error_curie_depth': float(rel_err_cd),
        'rel_error_beta': float(rel_err_beta),
        'rel_error_zt': float(rel_err_zt),
        'rel_error_dz': float(rel_err_dz),
        'mean_beta': float(mean_beta),
        'mean_zt': float(mean_zt),
        'mean_dz': float(mean_dz),
        'n_centroids_fitted': len(fit_cd),
    }
    
    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Magnetic anomaly grid
    ax = axes[0, 0]
    grid_data = grid_obj.data
    im = ax.imshow(grid_data, cmap='RdBu_r', aspect='equal', 
                    extent=[xmin/1e3, xmax/1e3, ymin/1e3, ymax/1e3], origin='lower')
    for xc_pt, yc_pt in centroids:
        ax.plot(xc_pt/1e3, yc_pt/1e3, 'k+', ms=15, mew=2)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Magnetic Anomaly Grid (nT)')
    plt.colorbar(im, ax=ax, label='nT')
    
    # (b) Radial power spectrum fit (centroid 1)
    ax = axes[0, 1]
    ax.plot(k, Phi_obs, 'ko', ms=3, alpha=0.5, label='Observed')
    ax.plot(k, Phi_true, 'b-', lw=2, alpha=0.7, label='True model')
    ax.plot(k, Phi_fit, 'r--', lw=2, label='Fitted model')
    ax.set_xlabel('Wavenumber (rad/km)')
    ax.set_ylabel('ln(Power)')
    ax.set_title('Radial Power Spectrum (Centroid 1)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (c) Curie depth bar chart across centroids
    ax = axes[0, 2]
    fit_cds = [r['curie_depth'] for r in fitted_results if not np.isnan(r['curie_depth'])]
    x_pos = range(len(fit_cds))
    ax.bar(x_pos, fit_cds, color='steelblue', alpha=0.7, label='Fitted')
    ax.axhline(true_curie, color='r', ls='--', lw=2, label=f'True ({true_curie:.1f} km)')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f'C{i+1}' for i in range(len(fit_cds))])
    ax.set_ylabel('Curie Depth (km)')
    ax.set_title('Curie Depth Recovery per Centroid')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # (d) Parameter recovery bar chart
    ax = axes[1, 0]
    param_names = ['β', 'z_t', 'Δz', 'z_Curie']
    rel_errors = [
        metrics['rel_error_beta'] * 100,
        metrics['rel_error_zt'] * 100,
        metrics['rel_error_dz'] * 100,
        metrics['rel_error_curie_depth'] * 100,
    ]
    bars = ax.bar(range(len(param_names)), rel_errors, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylabel('Mean Relative Error (%)')
    ax.set_title('Parameter Recovery Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rel_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # (e) Radial spectrum fit (centroid at center)
    ax = axes[1, 1]
    xc2, yc2 = centroids[-1]  # center centroid
    subgrid2 = grid_obj.subgrid(window_size, xc2, yc2)
    k2, Phi_obs2, _ = grid_obj.radial_spectrum(subgrid2, taper=np.hanning)
    fit_r2 = fitted_results[-1]
    Phi_fit2 = forward_operator(k2, fit_r2['beta'], fit_r2['zt'], fit_r2['dz'], fit_r2['C'])
    Phi_true2 = forward_operator(k2, true_beta, true_zt, true_dz, true_c)
    
    ax.plot(k2, Phi_obs2, 'ko', ms=3, alpha=0.5, label='Observed')
    ax.plot(k2, Phi_true2, 'b-', lw=2, alpha=0.7, label='True model')
    ax.plot(k2, Phi_fit2, 'r--', lw=2, label='Fitted model')
    ax.set_xlabel('Wavenumber (rad/km)')
    ax.set_ylabel('ln(Power)')
    ax.set_title('Radial Power Spectrum (Center)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (f) Parameter table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Parameter', 'True', 'Mean Fitted', 'Rel Err (%)'],
        ['β', f'{true_beta:.2f}', f'{metrics["mean_beta"]:.2f}', 
         f'{metrics["rel_error_beta"]*100:.1f}'],
        ['z_t (km)', f'{true_zt:.2f}', f'{metrics["mean_zt"]:.2f}',
         f'{metrics["rel_error_zt"]*100:.1f}'],
        ['Δz (km)', f'{true_dz:.2f}', f'{metrics["mean_dz"]:.2f}',
         f'{metrics["rel_error_dz"]*100:.1f}'],
        ['z_Curie (km)', f'{true_curie:.2f}', f'{metrics["mean_curie_depth_km"]:.2f}',
         f'{metrics["rel_error_curie_depth"]*100:.1f}'],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Parameter Recovery Summary')
    
    fig.suptitle(
        f"pycurious — Curie Point Depth Inversion\n"
        f"PSNR_spec={metrics['psnr_spectrum']:.2f} dB | CC_spec={metrics['cc_spectrum']:.4f} | "
        f"RMSE_CD={metrics['rmse_curie_depth_km']:.2f} km | RE_CD={metrics['rel_error_curie_depth']*100:.1f}%",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics
