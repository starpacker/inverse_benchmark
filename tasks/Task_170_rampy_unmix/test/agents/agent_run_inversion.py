import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import NMF

import rampy as rp

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def run_inversion(observed_spectra, wavenumber, n_components, pure_components):
    """
    Run the inverse solver: baseline correction followed by NMF unmixing.
    
    Parameters
    ----------
    observed_spectra : ndarray
        (n_mixtures, n_points) array of observed spectra.
    wavenumber : ndarray
        (n_points,) array of wavenumber values.
    n_components : int
        Number of components to extract.
    pure_components : ndarray
        (n_components, n_points) array of true pure components (for matching).
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - corrected_spectra: baseline-corrected spectra
        - fitted_baselines: estimated baselines
        - recovered_components: (n_components, n_points) recovered component spectra
        - recovered_weights: (n_mixtures, n_components) recovered mixing proportions
        - permutation: optimal component matching permutation
    """
    n_mixtures = observed_spectra.shape[0]
    n_points = observed_spectra.shape[1]
    
    # --- Step 1: Baseline correction using rampy ---
    corrected_spectra = np.zeros_like(observed_spectra)
    fitted_baselines = np.zeros_like(observed_spectra)
    
    for i in range(n_mixtures):
        y_corr, bl = rp.baseline(
            wavenumber, observed_spectra[i],
            method='als', lam=1e7, p=0.001, niter=100
        )
        corrected_spectra[i] = y_corr.flatten()
        fitted_baselines[i] = bl.flatten()
    
    # Clip negative values after baseline removal (physical constraint: intensities ≥ 0)
    corrected_spectra = np.clip(corrected_spectra, 0, None)
    
    print("Baseline correction completed.")
    
    # --- Step 2: NMF for spectral unmixing ---
    # NMF: V ≈ W * H, where V = (n_samples, n_features)
    # W = mixing proportions (n_samples, n_components)
    # H = component spectra (n_components, n_features)
    
    # Try multiple NMF initializations and pick the best
    best_err = np.inf
    best_W = None
    best_H = None
    best_seed = 0
    
    for seed in range(20):
        nmf_model = NMF(
            n_components=n_components,
            init='nndsvda',
            max_iter=10000,
            random_state=seed,
            l1_ratio=0.0,
            alpha_W=0.0,
            alpha_H=0.0,
            tol=1e-8,
        )
        W_try = nmf_model.fit_transform(corrected_spectra)
        H_try = nmf_model.components_
        err = nmf_model.reconstruction_err_
        if err < best_err:
            best_err = err
            best_W = W_try
            best_H = H_try
            best_seed = seed
    
    W_nmf = best_W  # (n_mixtures, n_components)
    H_nmf = best_H  # (n_components, n_points)
    
    print(f"NMF reconstruction error: {best_err:.6f} (seed={best_seed})")
    
    # --- Normalize NMF results ---
    # Normalize H rows to unit max, adjust W accordingly
    scale_factors = H_nmf.max(axis=1, keepdims=True)
    H_norm = H_nmf / scale_factors            # each component spectrum max = 1
    W_norm = W_nmf * scale_factors.T           # compensate in weights
    
    # Normalize W rows to sum to 1 (mixing proportions)
    W_sum = W_norm.sum(axis=1, keepdims=True)
    W_final = W_norm / W_sum
    
    # --- Match recovered components to true components ---
    # Use correlation-based optimal assignment (Hungarian algorithm)
    corr_matrix = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            corr_matrix[i, j] = np.corrcoef(pure_components[i], H_norm[j])[0, 1]
    
    # Maximize correlation → minimize negative correlation
    cost_matrix = -corr_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Permute recovered components and weights
    H_matched = H_norm[col_ind]
    W_matched = W_final[:, col_ind]
    
    print(f"Optimal permutation: true_idx -> recovered_idx = {list(zip(row_ind, col_ind))}")
    
    result = {
        'corrected_spectra': corrected_spectra,
        'fitted_baselines': fitted_baselines,
        'recovered_components': H_matched,
        'recovered_weights': W_matched,
        'permutation': col_ind,
    }
    
    return result
