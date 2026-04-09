import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

def evaluate_results(img_recon, expected_output_path, output_path, scaler, original_dtype):
    """
    Evaluate reconstruction results and save output.
    
    Parameters
    ----------
    img_recon : ndarray
        Reconstructed image (normalized).
    expected_output_path : str
        Path to expected output for comparison.
    output_path : str
        Path to save reconstructed image.
    scaler : float
        Scaling factor to restore original intensity range.
    original_dtype : dtype
        Original image data type for saving.
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, and MSE values.
    """
    # 1. Rescale output
    img_output = scaler * img_recon
    
    # 2. Save result
    io.imsave(output_path, img_output.astype(original_dtype))
    print(f"✅ Processing complete! Result saved to: {output_path}")
    
    # 3. Load expected output for comparison
    expected = io.imread(expected_output_path)
    
    # 4. Ensure shape matches
    if img_output.shape != expected.shape:
        raise ValueError("Reconstructed image and expected image must have the same shape!")
    
    # 5. Calculate metrics
    dynamic_range = expected.max() - expected.min()
    
    psnr = peak_signal_noise_ratio(expected, img_output, data_range=dynamic_range)
    
    c_axis = None if len(expected.shape) == 2 else 2
    ssim_val = ssim(expected, img_output, data_range=dynamic_range, channel_axis=c_axis)
    
    mse = np.mean((expected.astype(np.float64) - img_output.astype(np.float64)) ** 2)
    
    print(f"📊 PSNR: {psnr:.4f} dB")
    print(f"📊 SSIM: {ssim_val:.4f}")
    print(f"📊 MSE: {mse:.6f}")
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim_val,
        'mse': mse
    }
    
    return metrics