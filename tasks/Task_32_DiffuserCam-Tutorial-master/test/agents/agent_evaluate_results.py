import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

import os

def evaluate_results(recon, gt, result_name, output_dir="."):
    """
    Computes metrics if GT is available, and saves the image.
    """
    def min_max_scale(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    recon_norm = min_max_scale(recon)
    
    metrics_str = ""
    if gt is not None:
        gt_norm = min_max_scale(gt)
        # Use data_range=1.0 since we min-max scaled to [0,1]
        val_psnr = psnr(gt_norm, recon_norm, data_range=1.0)
        val_ssim = ssim(gt_norm, recon_norm, data_range=1.0)
        metrics_str = f"PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}"
        print(f"Evaluation for {result_name}: {metrics_str}")
    
    # Save Image
    plt.figure()
    plt.imshow(recon_norm, cmap='gray')
    title = f"{result_name}"
    if metrics_str:
        title += f"\n{metrics_str}"
    plt.title(title)
    plt.axis('off')
    
    out_path = os.path.join(output_dir, f"{result_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved result to {out_path}")
