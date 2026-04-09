import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import json

import os

def compute_metrics_linear(gt_map_2d, recon_map_2d):
    """Compute PSNR/SSIM on normalized linear-scale maps."""
    gt_max = np.max(gt_map_2d)
    if gt_max <= 0:
        gt_max = 1.0

    gt_n = gt_map_2d / gt_max
    recon_max = np.max(recon_map_2d)
    if recon_max <= 0:
        recon_n = np.zeros_like(recon_map_2d)
    else:
        recon_n = recon_map_2d / recon_max

    # Scale recon to minimize MSE (optimal scaling)
    scale = np.sum(gt_n * recon_n) / (np.sum(recon_n**2) + 1e-30)
    recon_scaled = np.clip(recon_n * scale, 0, 1)

    psnr = peak_signal_noise_ratio(gt_n, recon_scaled, data_range=1.0)
    ssim = structural_similarity(gt_n, recon_scaled, data_range=1.0)
    return psnr, ssim

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

def evaluate_results(q_gt, reconstructions, grid_res, results_dir, coords, mic_positions, params):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR and SSIM for each method, determines best method,
    saves results to files, and generates visualization.
    
    Parameters
    ----------
    q_gt : ndarray
        Ground truth source distribution (n_grid,)
    reconstructions : dict
        Dictionary of reconstructed source maps from run_inversion
    grid_res : int
        Grid resolution
    results_dir : str
        Directory to save results
    coords : ndarray
        1D array of grid coordinates
    mic_positions : ndarray
        Microphone positions (n_mics, 3)
    params : dict
        Dictionary of parameters (freq, n_mics, wavelength, snr_db, z_focus)
        
    Returns
    -------
    dict
        Metrics dictionary containing PSNR, SSIM for each method and best method info
    """
    os.makedirs(results_dir, exist_ok=True)
    
    gt_2d = q_gt.reshape(grid_res, grid_res)
    
    # Compute metrics for each method
    conv_2d = reconstructions['conventional'].reshape(grid_res, grid_res)
    clean_2d = reconstructions['clean_sc'].reshape(grid_res, grid_res)
    nnls_2d = reconstructions['nnls'].reshape(grid_res, grid_res)
    
    psnr_bf, ssim_bf = compute_metrics_linear(gt_2d, conv_2d)
    psnr_cl, ssim_cl = compute_metrics_linear(gt_2d, clean_2d)
    psnr_nn, ssim_nn = compute_metrics_linear(gt_2d, nnls_2d)
    
    print(f"  Conventional: PSNR={psnr_bf:.2f}dB, SSIM={ssim_bf:.4f}")
    print(f"  CLEAN-SC:     PSNR={psnr_cl:.2f}dB, SSIM={ssim_cl:.4f}")
    print(f"  NNLS:         PSNR={psnr_nn:.2f}dB, SSIM={ssim_nn:.4f}")
    
    # Determine best method
    results = {
        'conventional': {'psnr': psnr_bf, 'ssim': ssim_bf, 'map': reconstructions['conventional']},
        'clean_sc': {'psnr': psnr_cl, 'ssim': ssim_cl, 'map': reconstructions['clean_sc']},
        'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn, 'map': reconstructions['nnls']},
    }
    best_name = max(results, key=lambda m: results[m]['psnr'])
    best = results[best_name]
    
    print(f"\n  Best: {best_name} (PSNR={best['psnr']:.2f}dB, SSIM={best['ssim']:.4f})")
    
    # Save arrays
    recon_2d = best['map'].reshape(grid_res, grid_res)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_2d)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_2d)
    
    # Build metrics dictionary
    metrics = {
        'psnr_db': round(best['psnr'], 2),
        'ssim': round(best['ssim'], 4),
        'best_method': best_name,
        'conventional': {'psnr_db': round(psnr_bf, 2), 'ssim': round(ssim_bf, 4)},
        'clean_sc': {'psnr_db': round(psnr_cl, 2), 'ssim': round(ssim_cl, 4)},
        'nnls': {'psnr_db': round(psnr_nn, 2), 'ssim': round(ssim_nn, 4)},
        'parameters': {
            'frequency_hz': params['freq'],
            'n_mics': params['n_mics'],
            'grid_resolution': grid_res,
            'snr_db': params['snr_db'],
            'z_focus_m': params['z_focus'],
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plot
    plot_metrics = {
        'conv': {'psnr': psnr_bf, 'ssim': ssim_bf},
        'clean': {'psnr': psnr_cl, 'ssim': ssim_cl},
        'nnls': {'psnr': psnr_nn, 'ssim': ssim_nn}
    }
    
    plot_results(
        coords, gt_2d, conv_2d, clean_2d, nnls_2d,
        mic_positions, plot_metrics,
        os.path.join(results_dir, 'reconstruction_result.png'),
        params
    )
    
    return metrics
