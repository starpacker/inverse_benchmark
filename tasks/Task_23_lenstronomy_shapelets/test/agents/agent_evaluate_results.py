import numpy as np

def evaluate_results(
    inversion_result: dict,
    data_dict: dict
) -> dict:
    """
    Evaluate the results of the inversion.
    
    Computes metrics and returns summary statistics.
    """
    # Extract variables from inversion_result
    model_image = inversion_result['model_image']
    shapelet_coeffs = inversion_result['shapelet_coeffs']
    chi2_reduced = inversion_result['chi2_reduced']
    elapsed_time = inversion_result['elapsed_time']
    n_max = inversion_result['n_max']
    beta = inversion_result['beta']
    
    # Extract variables from data_dict
    image_data = data_dict['image_data']
    background_rms = data_dict['background_rms']
    
    # Compute residuals
    residuals = image_data - model_image
    
    # Compute RMS of residuals
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compute peak signal-to-noise ratio
    signal_max = np.max(np.abs(model_image))
    if residual_rms > 0:
        psnr = 20 * np.log10(signal_max / residual_rms)
    else:
        psnr = np.inf
    
    # Number of shapelet coefficients
    num_coeffs = len(shapelet_coeffs)
    
    # Expected number of coefficients for given n_max
    expected_num_coeffs = int((n_max + 1) * (n_max + 2) / 2)
    
    # Compute coefficient statistics
    coeff_mean = np.mean(shapelet_coeffs)
    coeff_std = np.std(shapelet_coeffs)
    coeff_max = np.max(np.abs(shapelet_coeffs))
    
    # Print evaluation results
    print(f"Reconstruction completed in {elapsed_time:.4f} seconds.")
    print(f"Reduced Chi^2: {chi2_reduced:.4f}")
    print(f"Number of Shapelet coefficients: {num_coeffs}")
    print(f"Expected coefficients for n_max={n_max}: {expected_num_coeffs}")
    print(f"Residual RMS: {residual_rms:.6f}")
    print(f"Background RMS: {background_rms:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Shapelet beta (scale): {beta}")
    print(f"Coefficient mean: {coeff_mean:.6f}")
    print(f"Coefficient std: {coeff_std:.6f}")
    print(f"Coefficient max abs: {coeff_max:.6f}")
    
    return {
        'residuals': residuals,
        'residual_rms': residual_rms,
        'psnr': psnr,
        'chi2_reduced': chi2_reduced,
        'num_coeffs': num_coeffs,
        'elapsed_time': elapsed_time,
        'coeff_mean': coeff_mean,
        'coeff_std': coeff_std,
        'coeff_max': coeff_max
    }