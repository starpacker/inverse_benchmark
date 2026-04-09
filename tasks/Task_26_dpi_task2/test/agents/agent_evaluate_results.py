import os
import sys
import numpy as np
import torch

# Ensure DPItorch is in path if specific modules are needed, 
# though this specific function relies primarily on standard libs.
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def evaluate_results(data, results, save_path):
    """
    Evaluate reconstruction results and save outputs.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        results: Dictionary from run_inversion
        save_path: Directory to save results
        
    Returns:
        Dictionary containing:
            - mean_reconstruction: Mean of reconstructed samples
            - rmse: Root mean squared error vs ground truth
            - psnr: Peak signal-to-noise ratio
    """
    # 1. Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_true = data['img_true']
    reconstructed = results['reconstructed']
    
    # 2. Compute mean reconstruction
    mean_reconstruction = np.mean(reconstructed, axis=0)
    
    # 3. Compute RMSE
    mse = np.mean((mean_reconstruction - img_true) ** 2)
    rmse = np.sqrt(mse)
    
    # 4. Compute PSNR
    max_val = np.max(img_true)
    if mse > 0:
        psnr = 20 * np.log10(max_val / rmse)
    else:
        psnr = float('inf')
    
    # 5. Save model and reconstruction
    torch.save(results['model'].state_dict(), os.path.join(save_path, 'mri_model.pth'))
    np.save(os.path.join(save_path, 'mri_reconstruction.npy'), reconstructed)
    np.save(os.path.join(save_path, 'mri_mean_reconstruction.npy'), mean_reconstruction)
    
    # 6. Print metrics
    print(f"Reconstruction RMSE: {rmse:.6f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    print(f"Saved model to {os.path.join(save_path, 'mri_model.pth')}")
    print(f"Saved reconstruction to {os.path.join(save_path, 'mri_reconstruction.npy')}")
    
    # 7. Return metrics
    return {
        'mean_reconstruction': mean_reconstruction,
        'rmse': rmse,
        'psnr': psnr
    }