import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import json

import os

def evaluate_results(
    T21_gt,
    residual_poly,
    residual_pca,
    observation,
    T_fg_mK,
    noise,
    frequencies,
    params,
    poly_order,
    n_pca_components,
    results_dir
):
    """
    Evaluate the inversion results with metrics and visualizations.
    
    Args:
        T21_gt: ground truth 21cm signal (n_freq x n_angle)
        residual_poly: recovered signal via polynomial fitting
        residual_pca: recovered signal via PCA
        observation: original observation
        T_fg_mK: foreground in mK
        noise: noise realization
        frequencies: frequency grid
        params: simulation parameters dict
        poly_order: polynomial order used
        n_pca_components: number of PCA components used
        results_dir: directory to save results
    
    Returns:
        dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    n_freq = params['n_freq']
    n_angle = params['n_angle']
    freq_min = params['freq_min']
    freq_max = params['freq_max']
    
    # Helper function: compute PSNR
    def compute_psnr(gt, recovered):
        data_range = np.max(gt) - np.min(gt)
        mse = np.mean((gt - recovered) ** 2)
        if mse == 0 or data_range == 0:
            return float('inf')
        return 10.0 * np.log10(data_range ** 2 / mse)
    
    # Helper function: compute correlation coefficient
    def compute_cc(gt, recovered):
        g = gt.ravel() - np.mean(gt)
        r = recovered.ravel() - np.mean(recovered)
        d = np.sqrt(np.sum(g**2) * np.sum(r**2))
        return float(np.sum(g * r) / d) if d > 0 else 0.0
    
    # Helper function: compute RMSE
    def compute_rmse(gt, recovered):
        return float(np.sqrt(np.mean((gt - recovered) ** 2)))
    
    # Compute metrics for polynomial method
    poly_psnr = compute_psnr(T21_gt, residual_poly)
    poly_cc = compute_cc(T21_gt, residual_poly)
    poly_rmse = compute_rmse(T21_gt, residual_poly)
    
    # Compute metrics for PCA method
    pca_psnr = compute_psnr(T21_gt, residual_pca)
    pca_cc = compute_cc(T21_gt, residual_pca)
    pca_rmse = compute_rmse(T21_gt, residual_pca)
    
    fg_ratio = np.std(T_fg_mK) / np.std(T21_gt)
    
    metrics = {
        'poly_psnr': float(poly_psnr),
        'poly_cc': float(poly_cc),
        'poly_rmse': float(poly_rmse),
        'pca_psnr': float(pca_psnr),
        'pca_cc': float(pca_cc),
        'pca_rmse': float(pca_rmse),
        'signal_rms_mK': float(np.std(T21_gt)),
        'foreground_signal_ratio': float(fg_ratio),
        'noise_rms_mK': float(np.std(noise)),
        'n_freq': n_freq,
        'n_angle': n_angle,
        'freq_range_MHz': [float(freq_min), float(freq_max)],
    }
    
    # Generate visualizations
    vmin_21 = np.percentile(T21_gt, 2)
    vmax_21 = np.percentile(T21_gt, 98)
    
    # Figure 1: Freq-Angle Maps
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('21cm Tomography: Foreground Removal Results', fontsize=16, y=0.98)
    
    kw = dict(aspect='auto', origin='lower',
              extent=[0, n_angle, freq_min, freq_max])
    
    im0 = axes[0, 0].imshow(T21_gt, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[0, 0].set_title('Ground Truth 21cm Signal')
    axes[0, 0].set_ylabel('Frequency (MHz)')
    axes[0, 0].set_xlabel('Angular Pixel')
    plt.colorbar(im0, ax=axes[0, 0], label='T (mK)')
    
    im1 = axes[0, 1].imshow(observation, cmap='inferno', **kw)
    axes[0, 1].set_title('Observation (Signal+FG+Noise)')
    axes[0, 1].set_ylabel('Frequency (MHz)')
    axes[0, 1].set_xlabel('Angular Pixel')
    plt.colorbar(im1, ax=axes[0, 1], label='T (mK)')
    
    im2 = axes[0, 2].imshow(T_fg_mK, cmap='inferno', **kw)
    axes[0, 2].set_title('Foreground (mK)')
    axes[0, 2].set_ylabel('Frequency (MHz)')
    axes[0, 2].set_xlabel('Angular Pixel')
    plt.colorbar(im2, ax=axes[0, 2], label='T (mK)')
    
    im3 = axes[1, 0].imshow(residual_poly, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[1, 0].set_title(f'Poly Fit (order={poly_order})\n'
                         f'PSNR={metrics["poly_psnr"]:.1f} dB, CC={metrics["poly_cc"]:.4f}')
    axes[1, 0].set_ylabel('Frequency (MHz)')
    axes[1, 0].set_xlabel('Angular Pixel')
    plt.colorbar(im3, ax=axes[1, 0], label='T (mK)')
    
    im4 = axes[1, 1].imshow(residual_pca, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[1, 1].set_title(f'PCA Recovery (n={n_pca_components})\n'
                         f'PSNR={metrics["pca_psnr"]:.1f} dB, CC={metrics["pca_cc"]:.4f}')
    axes[1, 1].set_ylabel('Frequency (MHz)')
    axes[1, 1].set_xlabel('Angular Pixel')
    plt.colorbar(im4, ax=axes[1, 1], label='T (mK)')
    
    error_pca = T21_gt - residual_pca
    im5 = axes[1, 2].imshow(error_pca, cmap='RdBu_r', **kw)
    axes[1, 2].set_title('Reconstruction Error (PCA)')
    axes[1, 2].set_ylabel('Frequency (MHz)')
    axes[1, 2].set_xlabel('Angular Pixel')
    plt.colorbar(im5, ax=axes[1, 2], label='ΔT (mK)')
    
    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, 'frequency_angle_maps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Figure 2: Spectral Profiles
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Spectral Profiles at Selected Angular Pixels', fontsize=14)
    pxs = [n_angle // 4, n_angle // 2, 3 * n_angle // 4, n_angle - 1]
    for ax, px in zip(axes2.ravel(), pxs):
        ax.plot(frequencies, T21_gt[:, px], 'k-', lw=2, label='GT 21cm', alpha=0.8)
        ax.plot(frequencies, residual_poly[:, px], 'b--', lw=1.5, label=f'Poly (ord={poly_order})', alpha=0.7)
        ax.plot(frequencies, residual_pca[:, px], 'r-.', lw=1.5, label=f'PCA (n={n_pca_components})', alpha=0.7)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Temperature (mK)')
        ax.set_title(f'Pixel {px}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, 'spectral_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Figure 3: Power Spectra
    from numpy.fft import rfft, rfftfreq
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Power Spectrum Analysis', fontsize=14)
    
    kf = rfftfreq(n_freq, d=(frequencies[1] - frequencies[0]))
    for data, label, ls in [(T21_gt, 'GT', 'k-'), (residual_poly, 'Poly', 'b--'), (residual_pca, 'PCA', 'r-.')]:
        ax3a.semilogy(kf[1:], np.mean(np.abs(rfft(data, axis=0))**2, axis=1)[1:], ls, lw=1.5, label=label)
    ax3a.set_xlabel('Freq mode (1/MHz)')
    ax3a.set_ylabel('Power')
    ax3a.set_title('Frequency Power Spectrum')
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)
    
    ka = rfftfreq(n_angle)
    for data, label, ls in [(T21_gt, 'GT', 'k-'), (residual_poly, 'Poly', 'b--'), (residual_pca, 'PCA', 'r-.')]:
        ax3b.semilogy(ka[1:], np.mean(np.abs(rfft(data, axis=1))**2, axis=0)[1:], ls, lw=1.5, label=label)
    ax3b.set_xlabel('Angular mode')
    ax3b.set_ylabel('Power')
    ax3b.set_title('Angular Power Spectrum')
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, 'power_spectra.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Grid:              {n_freq} freq × {n_angle} angle")
    print(f"  Freq range:        {freq_min}–{freq_max} MHz")
    print(f"  Signal RMS:        {np.std(T21_gt):.2f} mK")
    print(f"  FG/Signal ratio:   {fg_ratio:.0f}x")
    print(f"  Noise RMS:         {np.std(noise):.2f} mK")
    print(f"  ── Polynomial (order={poly_order}) ──")
    print(f"     PSNR = {poly_psnr:.2f} dB | CC = {poly_cc:.4f} | RMSE = {poly_rmse:.2f} mK")
    print(f"  ── PCA (n_comp={n_pca_components}) ──")
    print(f"     PSNR = {pca_psnr:.2f} dB | CC = {pca_cc:.4f} | RMSE = {pca_rmse:.2f} mK")
    print("=" * 70)
    print(f"  Figures saved to {results_dir}")
    print(f"  Metrics saved to {metrics_path}")
    
    return metrics
