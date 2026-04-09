import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def to_db(source_map, dynamic_range=30.0):
    """Convert to dB with dynamic range."""
    mx = np.max(source_map)
    if mx <= 0:
        return np.full_like(source_map, -dynamic_range)
    n = np.maximum(source_map / mx, 10 ** (-dynamic_range / 10))
    return 10.0 * np.log10(n)

def plot_results(coords, gt_2d, bf_2d, clean_2d, nnls_2d,
                 mic_pos, metrics_dict, save_path, params):
    """4-panel plot."""
    extent = [coords[0], coords[-1], coords[0], coords[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (axes[0, 0], gt_2d, 'Ground Truth Source Distribution', None),
        (axes[0, 1], bf_2d, 'Conventional Beamforming',
         f"PSNR={metrics_dict['conv']['psnr']:.2f}dB, SSIM={metrics_dict['conv']['ssim']:.4f}"),
        (axes[1, 0], clean_2d, 'CLEAN-SC Deconvolution',
         f"PSNR={metrics_dict['clean']['psnr']:.2f}dB, SSIM={metrics_dict['clean']['ssim']:.4f}"),
        (axes[1, 1], nnls_2d, 'NNLS Inversion',
         f"PSNR={metrics_dict['nnls']['psnr']:.2f}dB, SSIM={metrics_dict['nnls']['ssim']:.4f}"),
    ]

    for ax, data, title, subtitle in panels:
        db = to_db(data)
        im = ax.imshow(db, extent=extent, origin='lower', cmap='hot',
                       vmin=-30, vmax=0, aspect='equal')
        full_title = title if subtitle is None else f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=12, fontweight='bold' if subtitle is None else 'normal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(im, ax=ax, label='Power [dB]')
        ax.scatter(mic_pos[:, 0], mic_pos[:, 1],
                   c='cyan', s=8, alpha=0.5, marker='.', zorder=5)

    freq = params['freq']
    n_mics = params['n_mics']
    wavelength = params['wavelength']
    snr_db = params['snr_db']
    
    plt.suptitle(f'Acoustic Beamforming: Source Localization\n'
                 f'({n_mics} mics, f={freq:.0f}Hz, λ={wavelength*100:.1f}cm, SNR={snr_db:.0f}dB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")
