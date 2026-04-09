import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

def evaluate_results(data, result, results_dir=None):
    """
    Evaluate reconstruction quality and optionally save results.
    
    Parameters
    ----------
    data : dict - output from load_and_preprocess_data
    result : dict - output from run_inversion
    results_dir : str or None - directory to save results (if None, skip saving)
    
    Returns
    -------
    metrics : dict - evaluation metrics
    """
    from skimage.metrics import structural_similarity as ssim_func
    
    model = data['model']
    final_image = result['final_image']
    dirty_image = result['dirty_image']
    
    # Compute PSNR
    def compute_psnr(ref, test, data_range=None):
        if data_range is None:
            data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float))**2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range**2 / mse)
    
    # Compute SSIM
    def compute_ssim(ref, test):
        data_range = ref.max() - ref.min()
        return ssim_func(ref, test, data_range=data_range)
    
    # Compute correlation coefficient
    def compute_cc(ref, test):
        return float(np.corrcoef(ref.ravel(), test.ravel())[0, 1])
    
    n_valid = int(np.sum(data['valid']))
    
    metrics = {
        "task": "suncasa_radio",
        "task_id": 199,
        "method": result['method_used'],
        "n_antennas": data['config']['n_ant'],
        "n_visibilities": n_valid,
        "image_size": data['config']['n'],
        "clean_iterations": len(result['clean_components']),
        "psnr": float(compute_psnr(model, final_image)),
        "ssim": float(compute_ssim(model, final_image)),
        "cc": float(compute_cc(model, final_image)),
        "rmse": float(np.sqrt(np.mean((model - final_image)**2))),
        "dirty_psnr": float(compute_psnr(model, dirty_image)),
        "clean_psnr": float(result['psnr_clean']),
        "wiener_psnr": float(result['psnr_wiener']),
    }
    
    print(f"  PSNR = {metrics['psnr']:.2f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")
    print(f"  CC   = {metrics['cc']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  Dirty PSNR = {metrics['dirty_psnr']:.2f} dB (baseline)")
    
    # Save results if directory provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")
        
        np.save(os.path.join(results_dir, "ground_truth.npy"), model)
        np.save(os.path.join(results_dir, "reconstruction.npy"), final_image)
        
        # Visualization
        u = data['u']
        v = data['v']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Task 199: Solar Radio Image Reconstruction (CLEAN)\n"
            f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | CC={metrics['cc']:.4f}",
            fontsize=14, fontweight='bold'
        )
        
        vmin, vmax = 0, model.max()
        
        # Row 1: Images
        ax = axes[0, 0]
        im = ax.imshow(model, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('Ground Truth\n(Solar Radio Model)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[0, 1]
        im = ax.imshow(dirty_image, cmap='hot', origin='lower')
        ax.set_title('Dirty Image\n(with sidelobes)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[0, 2]
        im = ax.imshow(final_image, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('CLEAN Reconstruction')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: UV coverage, error map, profiles
        ax = axes[1, 0]
        ax.scatter(u, v, s=0.1, c='blue', alpha=0.3)
        ax.set_xlabel('u (wavelengths)')
        ax.set_ylabel('v (wavelengths)')
        ax.set_title(f'(u,v) Coverage\n({len(u)} visibilities)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        error = np.abs(model - final_image)
        im = ax.imshow(error, cmap='viridis', origin='lower')
        ax.set_title('Error Map\n|GT - CLEAN|')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[1, 2]
        mid = model.shape[0] // 2
        ax.plot(model[mid, :], 'b-', lw=2, label='Ground Truth')
        ax.plot(dirty_image[mid, :], 'gray', alpha=0.5, label='Dirty')
        ax.plot(final_image[mid, :], 'r--', lw=2, label='CLEAN')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Brightness')
        ax.set_title('Central Profile Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = os.path.join(results_dir, "reconstruction_result.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIS] Saved → {vis_path}")
    
    return metrics
