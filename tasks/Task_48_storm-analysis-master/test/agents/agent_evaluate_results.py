import numpy as np

def evaluate_results(original, reconstructed):
    """
    Calculates PSNR.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    data_range = np.max(original) - np.min(original)
    if data_range == 0:
        return 0.0
        
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr
