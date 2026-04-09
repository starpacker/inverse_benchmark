import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_results(data_dict, result_dict, save_dir=None):
    """
    Evaluate and visualize the inversion results.
    
    Computes final metrics, saves results to disk, and generates visualization.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        result_dict: Dictionary from run_inversion
        save_dir: Directory to save results (default: ./results)
        
    Returns:
        metrics: Dictionary containing final PSNR, SSIM, and other metadata
    """
    gt_u0 = data_dict['gt_u0']
    u_obs = data_dict['u_obs']
    u_T = data_dict['u_T']
    params = data_dict['params']
    
    recon = result_dict['best_recon']
    loss_history = result_dict['loss_history']
    psnr_history = result_dict['psnr_history']
    
    # Compute final metrics
    gt_np = gt_u0.detach().cpu().numpy().astype(np.float64)
    recon_np = recon.detach().cpu().numpy().astype(np.float64)
    data_range = max(gt_np.max() - gt_np.min(), 1e-10)
    
    final_psnr = psnr(gt_np, recon_np, data_range=data_range)
    final_ssim = ssim(gt_np, recon_np, data_range=data_range)
    
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"  PSNR: {final_psnr:.2f} dB")
    print(f"  SSIM: {final_ssim:.4f}")
    print(f"{'='*50}")
    
    # Setup save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        "psnr": round(float(final_psnr), 2),
        "ssim": round(float(final_ssim), 4),
        "noise_level": params['noise_level'],
        "n_iters": len(loss_history),
        "grid_size": [params['nx'], params['ny']],
        "alpha": params['alpha'],
        "n_steps": params['n_steps'],
        "method": "differentiable_pde_inversion_pytorch_autograd"
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")
    
    # Save arrays
    obs_np = u_obs.cpu().numpy()
    u_T_np = u_T.cpu().numpy()
    
    np.save(os.path.join(save_dir, 'ground_truth.npy'), gt_np)
    np.save(os.path.join(save_dir, 'reconstruction.npy'), recon_np)
    np.save(os.path.join(save_dir, 'observation.npy'), obs_np)
    print(f"Saved .npy arrays")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Fields
    vmin = min(gt_np.min(), recon_np.min())
    vmax = max(gt_np.max(), recon_np.max())
    
    im0 = axes[0, 0].imshow(gt_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth u₀', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(recon_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Recovered u₀ (PSNR={final_psnr:.1f}dB)', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    diff = np.abs(gt_np - recon_np)
    im2 = axes[0, 2].imshow(diff.T, origin='lower', cmap='viridis')
    axes[0, 2].set_title(f'|Error| (max={diff.max():.4f})', fontsize=14)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Observation and convergence
    im3 = axes[1, 0].imshow(obs_np.T, origin='lower', cmap='hot')
    axes[1, 0].set_title('Noisy Observation u(T)+noise', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(u_T_np.T, origin='lower', cmap='hot')
    axes[1, 1].set_title('Clean u(T) (forward solution)', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    axes[1, 2].semilogy(loss_history)
    axes[1, 2].set_title('Optimization Loss', fontsize=14)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('MSE Loss')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add PSNR convergence on twin axis
    if psnr_history:
        ax_twin = axes[1, 2].twinx()
        iters, psnrs = zip(*psnr_history)
        ax_twin.plot(iters, psnrs, 'r-o', markersize=3, label='PSNR')
        ax_twin.set_ylabel('PSNR (dB)', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
    
    nx, ny = params['nx'], params['ny']
    alpha = params['alpha']
    
    plt.suptitle(
        f'Heat Equation Initial Condition Inversion\n'
        f'∂u/∂t = α∇²u, α={alpha}, Grid={nx}×{ny}, '
        f'PSNR={final_psnr:.2f}dB, SSIM={final_ssim:.4f}',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction_result.png")
    
    print(f"\nAll results saved to {save_dir}/")
    
    return metrics
