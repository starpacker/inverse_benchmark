import numpy as np

def evaluate_results(original_data, reconstructed_data):
    """
    Computes error metrics between original and reconstructed data.
    """
    # Create mask for valid data (exclude NaNs from thresholding)
    mask = ~np.isnan(reconstructed_data) & ~np.isnan(original_data)
    
    if mask.sum() == 0:
        return {'mse': float('nan'), 'psnr': float('nan'), 'corr': float('nan')}
        
    orig_valid = original_data[mask]
    rec_valid = reconstructed_data[mask]
    
    # MSE
    mse = np.mean((orig_valid - rec_valid)**2)
    
    # PSNR
    max_val = np.max(orig_valid)
    if mse > 0:
        psnr = 10 * np.log10(max_val**2 / mse)
    else:
        psnr = float('inf')
        
    # Correlation Coefficient
    if orig_valid.size > 1:
        corr = np.corrcoef(orig_valid.flatten(), rec_valid.flatten())[0, 1]
    else:
        corr = 0.0
        
    return {'mse': mse, 'psnr': psnr, 'corr': corr}
