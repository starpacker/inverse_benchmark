import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_results(result_img, reference_img):
    """
    Calculates PSNR and RMSE, and saves a comparison plot.
    """
    # Calculate Metrics
    mse = np.mean((result_img - reference_img) ** 2)
    if mse == 0:
        psnr = float('inf')
        rmse = 0.0
    else:
        data_range = 255.0 # Since images are uint8
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        rmse = np.sqrt(mse)
    
    print(f"\nEvaluation Results:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"RMSE: {rmse:.2f}")
    
    return psnr, rmse
