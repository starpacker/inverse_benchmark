import numpy as np

import matplotlib.pyplot as plt

def calculate_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return 100
    max_pixel = gt.max()
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(gt, recon):
    """Structural Similarity Index Wrapper"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)
    except ImportError:
        return 0

def norm_minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def evaluate_results(gt, recon_dict):
    """
    Calculates metrics and generates visualization.
    gt: Ground Truth Image
    recon_dict: Dictionary {method_name: reconstructed_image}
    """
    gt_norm = norm_minmax(gt)
    
    # Create circular mask
    h, w = gt_norm.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - w/2)**2 + (y - h/2)**2 <= (w/2)**2
    
    results_text = []
    
    for name, recon in recon_dict.items():
        r_norm = norm_minmax(recon)
        psnr = calculate_psnr(gt_norm[mask], r_norm[mask])
        ssim = calculate_ssim(gt_norm, r_norm)
        results_text.append(f"{name} -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        print(results_text[-1])

    # Visualization
    try:
        num_plots = 1 + len(recon_dict)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1: axes = [axes] # Handle single plot case if empty dict
        
        # Plot GT
        axes[0].imshow(gt, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Plot Recons
        for i, (name, recon) in enumerate(recon_dict.items(), 1):
            axes[i].imshow(recon, cmap='gray')
            axes[i].set_title(name)
            axes[i].axis('off')
            
        output_file = 'tomopy_workflow_refactored.png'
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        
    return results_text
