import numpy as np

import matplotlib

matplotlib.use('Agg')

def run_inversion(observation, frequencies, freq_ref, poly_order, n_pca_components):
    """
    Run foreground removal inversion using both polynomial fitting and PCA.
    
    Args:
        observation: observed brightness temperature (n_freq x n_angle) in mK
        frequencies: frequency grid array in MHz
        freq_ref: reference frequency in MHz
        poly_order: polynomial order for log-frequency fitting
        n_pca_components: number of PCA/SVD modes to remove
    
    Returns:
        dict containing:
            - residual_poly: recovered 21cm signal via polynomial fitting
            - fg_estimate_poly: estimated foreground via polynomial fitting
            - residual_pca: recovered 21cm signal via PCA
            - fg_estimate_pca: estimated foreground via PCA
    """
    n_freq, n_angle = observation.shape
    
    # Method 1: Polynomial foreground removal
    # Fit & subtract polynomial in log(ν) space per angular pixel.
    # Power-law foregrounds ∝ ν^{-β} are nearly polynomial in log(ν).
    residual_poly = np.zeros_like(observation)
    fg_estimate_poly = np.zeros_like(observation)
    log_freq = np.log(frequencies / freq_ref)
    
    for j in range(n_angle):
        coeffs = np.polyfit(log_freq, observation[:, j], poly_order)
        poly_fit = np.polyval(coeffs, log_freq)
        fg_estimate_poly[:, j] = poly_fit
        residual_poly[:, j] = observation[:, j] - poly_fit
    
    # Method 2: PCA/SVD foreground removal
    # Leading singular modes capture the spectrally smooth foreground.
    # Subtracting them reveals the 21cm signal.
    U, S, Vt = np.linalg.svd(observation, full_matrices=False)
    fg_estimate_pca = np.zeros_like(observation)
    for k in range(n_pca_components):
        fg_estimate_pca += S[k] * np.outer(U[:, k], Vt[k, :])
    residual_pca = observation - fg_estimate_pca
    
    return {
        'residual_poly': residual_poly,
        'fg_estimate_poly': fg_estimate_poly,
        'residual_pca': residual_pca,
        'fg_estimate_pca': fg_estimate_pca
    }
