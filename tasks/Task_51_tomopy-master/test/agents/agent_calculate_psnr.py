import numpy as np

def calculate_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return 100
    max_pixel = gt.max()
    return 20 * np.log10(max_pixel / np.sqrt(mse))
