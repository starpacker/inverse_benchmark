import numpy as np

from scipy.ndimage import shift, gaussian_filter

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.registration import phase_cross_correlation

def evaluate_results(result, data_container):
    """
    Compares reconstruction to ground truth if available and prints metrics.
    
    Args:
        result (dict): Output from run_inversion.
        data_container (dict): Original data dictionary.
        
    Returns:
        tuple: (PSNR, SSIM)
    """
    recon_obj = result['reconstructed_object']
    gt_obj = data_container.get('ground_truth_object')
    positions = data_container['positions']
    No = data_container['No']
    Np = data_container['Np']
    
    if gt_obj is None:
        print("No Ground Truth available for evaluation.")
        return 0.0, 0.0
    
    recon_amp = np.abs(recon_obj)
    gt_amp = np.abs(gt_obj)
    
    print("Evaluating results...")
    
    # 1. Registration (Translation correction)
    # Using subpixel registration on amplitude
    shift_vector, error, diffphase = phase_cross_correlation(gt_amp, recon_amp, upsample_factor=10)
    print(f"  Detected shift: {shift_vector}")
    
    # Apply shift to the complex object
    recon_aligned = shift(recon_obj, shift_vector, mode='wrap')
    recon_amp_aligned = np.abs(recon_aligned)
    
    # 2. ROI Selection (Focus on illuminated area to avoid background noise bias)
    min_r, min_c = np.min(positions, axis=0)
    max_r, max_c = np.max(positions, axis=0)
    roi_slice = (
        slice(max(0, min_r), min(No, max_r + Np)),
        slice(max(0, min_c), min(No, max_c + Np))
    )
    
    recon_roi = recon_amp_aligned[roi_slice]
    gt_roi = gt_amp[roi_slice]
    
    # 3. Scale Matching (Linear Regression y = ax)
    # Match reconstruction intensity to GT intensity
    numerator = np.sum(recon_roi * gt_roi)
    denominator = np.sum(recon_roi**2)
    if denominator < 1e-10: denominator = 1e-10
    scale_opt = numerator / denominator
    
    recon_roi_scaled = recon_roi * scale_opt
    
    # 4. Normalize for Metrics
    max_val = np.max(gt_roi)
    if max_val < 1e-10: max_val = 1.0
    
    recon_final = np.clip(recon_roi_scaled, 0, max_val) / max_val
    gt_final = gt_roi / max_val
    
    # 5. Compute Metrics
    p_val = psnr(gt_final, recon_final, data_range=1.0)
    s_val = ssim(gt_final, recon_final, data_range=1.0)
    
    print(f"  Optimal Scale Factor: {scale_opt:.4f}")
    print(f"  PSNR: {p_val:.2f} dB")
    print(f"  SSIM: {s_val:.4f}")
    
    return p_val, s_val
