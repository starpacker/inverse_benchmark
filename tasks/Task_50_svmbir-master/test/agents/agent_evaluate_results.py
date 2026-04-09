import numpy as np

import matplotlib.pyplot as plt

try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

HAS_SKIMAGE = True

def evaluate_results(gt, recon, save_path="reconstruction_result.png"):
    """
    Computes PSNR/SSIM and saves a comparison plot.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    # Normalize for fair metric calculation
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn == 0: return arr
        return (arr - mn) / (mx - mn)

    gt_norm = normalize(gt)
    recon_norm = normalize(recon)
    
    # PSNR
    mse = np.mean((gt_norm - recon_norm) ** 2)
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
        
    # SSIM
    ssim_val = 0.0
    if HAS_SKIMAGE:
        ssim_val = ssim_func(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Evaluation -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    
    # Visualization
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt, cmap='gray')
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
        
        ax[1].imshow(recon, cmap='gray')
        ax[1].set_title(f"Reconstruction\nPSNR: {psnr_val:.1f}")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Figure saved to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    return {"psnr": psnr_val, "ssim": ssim_val}
