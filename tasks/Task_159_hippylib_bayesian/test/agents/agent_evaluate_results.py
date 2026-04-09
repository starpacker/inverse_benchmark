import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def compute_psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon)**2)
    if mse < 1e-30:
        return 100.0
    data_range = np.max(x_true) - np.min(x_true)
    return 20.0 * np.log10(data_range / np.sqrt(mse))

def compute_cc(x_true, x_recon):
    return float(np.corrcoef(x_true.ravel(), x_recon.ravel())[0, 1])

def compute_relative_error(x_true, x_recon):
    return float(np.linalg.norm(x_true - x_recon) / (np.linalg.norm(x_true) + 1e-30))

def compute_ssim_simple(x_true, x_recon):
    mu_x, mu_y = np.mean(x_true), np.mean(x_recon)
    sigma_x, sigma_y = np.std(x_true), np.std(x_recon)
    sigma_xy = np.mean((x_true - mu_x) * (x_recon - mu_y))
    data_range = np.max(x_true) - np.min(x_true)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2*mu_x*mu_y+c1)*(2*sigma_xy+c2)) /
                 ((mu_x**2+mu_y**2+c1)*(sigma_x**2+sigma_y**2+c2)))

def evaluate_results(data, inversion_result, results_dir=None):
    """
    Evaluate reconstruction quality and optionally save results.
    
    Args:
        data: dict containing preprocessed data including ground truth
        inversion_result: dict containing optimized parameters and reconstruction
        results_dir: optional path to save results
        
    Returns:
        dict containing all computed metrics
    """
    kappa_true = data['kappa_true']
    kappa_map = inversion_result['kappa_map']
    params_opt = inversion_result['params_opt']
    x = data['x']
    y = data['y']
    obs_ix = data['obs_ix']
    obs_iy = data['obs_iy']
    N_grid = data['N_grid']
    n_obs = data['n_obs']
    n_sources = data['n_sources']
    n_basis = data['n_basis']
    snr_db = data['snr_db']
    
    # Compute metrics
    psnr = compute_psnr(kappa_true, kappa_map)
    cc = compute_cc(kappa_true, kappa_map)
    re = compute_relative_error(kappa_true, kappa_map)
    ssim = compute_ssim_simple(kappa_true, kappa_map)
    
    print(f"\n=== Final Results ===")
    print(f"PSNR:  {psnr:.2f} dB")
    print(f"CC:    {cc:.4f}")
    print(f"RE:    {re:.4f} ({re*100:.2f}%)")
    print(f"SSIM:  {ssim:.4f}")
    print(f"MAP kappa: [{kappa_map.min():.3f}, {kappa_map.max():.3f}]")
    print(f"Params: {params_opt}")
    
    metrics = {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "cc": float(cc),
        "relative_error": float(re),
        "n_grid": N_grid,
        "n_obs": n_obs,
        "n_sources": n_sources,
        "n_basis": n_basis,
        "snr_db": snr_db,
    }
    
    # Save results if directory provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        
        np.save(os.path.join(results_dir, "ground_truth.npy"), kappa_true)
        np.save(os.path.join(results_dir, "reconstruction.npy"), kappa_map)
        
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Visualization
        error_map = np.abs(kappa_true - kappa_map)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        vmin_k = min(kappa_true.min(), kappa_map.min())
        vmax_k = max(kappa_true.max(), kappa_map.max())
        
        im0 = axes[0,0].imshow(kappa_true, origin='lower', extent=[0,1,0,1],
                                cmap='viridis', vmin=vmin_k, vmax=vmax_k)
        axes[0,0].set_title(r'True $\kappa(x)$', fontsize=14)
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0,0], fraction=0.046)
        
        im1 = axes[0,1].imshow(kappa_map, origin='lower', extent=[0,1,0,1],
                                cmap='viridis', vmin=vmin_k, vmax=vmax_k)
        axes[0,1].set_title(r'MAP Estimate $\kappa_{MAP}(x)$', fontsize=14)
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,1], fraction=0.046)
        axes[0,1].scatter(x[obs_ix], y[obs_iy], c='red', s=8, marker='x', alpha=0.5)
        
        im2 = axes[1,0].imshow(error_map, origin='lower', extent=[0,1,0,1], cmap='hot')
        axes[1,0].set_title(r'$|\kappa_{true} - \kappa_{MAP}|$', fontsize=14)
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1,0], fraction=0.046)
        
        mid = N_grid // 2
        axes[1,1].plot(x, kappa_true[mid,:], 'b-', lw=2, label='True')
        axes[1,1].plot(x, kappa_map[mid,:], 'r--', lw=2, label='MAP')
        axes[1,1].set_title(f'Cross-section y={y[mid]:.2f}', fontsize=14)
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel(r'$\kappa$')
        axes[1,1].legend(fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        fig.suptitle(f'PDE-Constrained Bayesian Inversion\n'
                     f'PSNR={psnr:.1f}dB, CC={cc:.3f}, RE={re*100:.1f}%, SSIM={ssim:.3f}',
                     fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nVisualization saved to {results_dir}")
    
    return metrics
