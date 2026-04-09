import numpy as np

import cv2

from math import pi, sqrt, log10

def calculate_psnr(img1, img2):
    """Calculates PSNR between two normalized images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two images.
    """
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Gaussian kernel for local mean
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def evaluate_results(reconstructed_field, gt_amp, gt_phase):
    """
    Calculates metrics and saves images.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    reconstructed_amplitude = np.abs(reconstructed_field)
    reconstructed_phase = np.angle(reconstructed_field)
    
    # Clip result to valid range [0, 1] for amplitude comparison
    reconstructed_amplitude_clipped = np.clip(reconstructed_amplitude, 0, 1)
    
    psnr_val = calculate_psnr(gt_amp, reconstructed_amplitude_clipped)
    ssim_val = calculate_ssim(gt_amp, reconstructed_amplitude_clipped)
    
    print(f"Amplitude PSNR: {psnr_val:.2f} dB")
    print(f"Amplitude SSIM: {ssim_val:.4f}")
    
    # Save outputs
    cv2.imwrite('output_gt_amp.png', (gt_amp * 255).astype(np.uint8))
    cv2.imwrite('output_reconstruction_amp.png', (reconstructed_amplitude_clipped * 255).astype(np.uint8))
    
    # Visualize Phase (Normalize to 0-255 for visualization)
    gt_phase_norm = ((gt_phase - gt_phase.min()) / (gt_phase.max() - gt_phase.min()) * 255).astype(np.uint8)
    rec_phase_norm = ((reconstructed_phase - reconstructed_phase.min()) / (reconstructed_phase.max() - reconstructed_phase.min()) * 255).astype(np.uint8)
    
    cv2.imwrite('output_gt_phase.png', gt_phase_norm)
    cv2.imwrite('output_reconstruction_phase.png', rec_phase_norm)
    
    print("Saved output images: output_gt_amp.png, output_reconstruction_amp.png, output_gt_phase.png, output_reconstruction_phase.png")
    
    return {"PSNR": psnr_val, "SSIM": ssim_val}
