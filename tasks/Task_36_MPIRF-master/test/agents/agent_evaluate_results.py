import numpy as np

def evaluate_results(ground_truth, reconstructed_image):
    """
    Compares the ground truth phantom with the reconstructed image.
    Calculates PSNR and SSIM.
    """
    
    # Ground truth also needs to be cropped to match the reconstruction
    # Reconstruction logic: c[1:-1, 1:-1]
    gt_cropped = ground_truth[1:-1, 1:-1]
    
    # Normalize Ground Truth
    max_gt = np.max(gt_cropped)
    if max_gt > 0:
        gt_norm = gt_cropped / max_gt
    else:
        gt_norm = gt_cropped

    target = gt_norm
    prediction = reconstructed_image
    
    data_range = target.max() - target.min()
    if data_range == 0: 
        data_range = 1.0

    # --- PSNR ---
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse))

    # --- SSIM (Simplified) ---
    mu_x = target.mean()
    mu_y = prediction.mean()
    var_x = target.var()
    var_y = prediction.var()
    cov_xy = np.mean((target - mu_x) * (prediction - mu_y))
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    
    ssim_val = numerator / denominator
    
    print(f"Evaluation Metrics:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MSE': mse
    }
