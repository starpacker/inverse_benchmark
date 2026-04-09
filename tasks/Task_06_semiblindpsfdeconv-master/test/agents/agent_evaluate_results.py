import logging

from skimage import io, metrics, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate_results(ground_truth, reconstructed):
    """
    Computes PSNR and SSIM.
    """
    psnr_val = metrics.peak_signal_noise_ratio(ground_truth, reconstructed, data_range=1.0)
    ssim_val = metrics.structural_similarity(ground_truth, reconstructed, data_range=1.0)
    
    return psnr_val, ssim_val
