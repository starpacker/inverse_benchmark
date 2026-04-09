import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

def evaluate_results(reconstruction, gt_images):
    """
    Computes PSNR and SSIM if ground truth is available.
    
    Args:
        reconstruction: (2, H, W) numpy array.
        gt_images: (2, H, W) numpy array or None.
    """
    if gt_images is None:
        print("Ground truth not available for quantitative evaluation.")
        return

    # Clip negative values
    res_images = np.maximum(reconstruction, 0)
    
    metrics = []
    material_names = ["Bone/Calcium", "Soft Tissue/Water"]
    
    print("\n=== Evaluation ===")
    for i in range(2):
        gt = gt_images[i]
        rec = res_images[i]
        
        # Normalize both to [0,1] for fair comparison (handles unit mismatch)
        gt_min, gt_max = np.min(gt), np.max(gt)
        rec_min, rec_max = np.min(rec), np.max(rec)
        denom_gt = gt_max - gt_min if gt_max != gt_min else 1.0
        denom_rec = rec_max - rec_min if rec_max != rec_min else 1.0
        gt = (gt - gt_min) / denom_gt
        rec = (rec - rec_min) / denom_rec
        
        # Dynamic range for PSNR (now both in [0,1])
        dmax = 1.0
        
        p = psnr(gt, rec, data_range=dmax)
        s = ssim(gt, rec, data_range=dmax)
        metrics.append((p, s))
        
        print(f"Material {i+1} ({material_names[i]}): PSNR = {p:.2f} dB, SSIM = {s:.4f}")
        
    avg_psnr = np.mean([m[0] for m in metrics])
    avg_ssim = np.mean([m[1] for m in metrics])
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
