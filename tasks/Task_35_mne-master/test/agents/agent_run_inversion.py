import numpy as np

import scipy.linalg

def run_inversion(G, C, y, nave, P, snr=3.0, method='dSPM'):
    """
    Computes the inverse operator K and applies it to sensor data y.
    
    Mathematical Steps:
    1. C_scaled = C / nave
    2. Project C and G using P
    3. Whiten G -> G_tilde
    4. SVD of G_tilde
    5. Compute K
    6. Apply K to y
    7. Apply noise normalization (if dSPM)
    
    Returns:
        source_estimate (np.ndarray): (n_sources, n_times)
    """
    print("\n--- [Standalone] Fitting Inverse Model ---")
    lambda2 = 1.0 / (snr ** 2)
    
    # 0. Adjust Noise Covariance for Averaging
    C_scaled = C / nave
    
    # 1. Apply Projections
    # C_proj = P * C_scaled * P^T
    C_proj = np.dot(P, np.dot(C_scaled, P.T))
    # G_proj = P * G
    G_proj = np.dot(P, G)
    
    # 2. Compute Whitener W = C^(-1/2) using eigh
    eig_vals, eig_vecs = scipy.linalg.eigh(C_proj)
    
    # Filter small eigenvalues
    max_eig = np.max(eig_vals)
    tol = 1e-6 * max_eig
    mask = eig_vals > tol
    
    print(f"Rank of Noise Covariance: {np.sum(mask)} / {len(eig_vals)}")
    
    eig_vals = eig_vals[mask]
    eig_vecs = eig_vecs[:, mask]
    
    # W = Lambda^(-1/2) V^T
    W = np.dot(eig_vecs, np.dot(np.diag(1.0 / np.sqrt(eig_vals)), eig_vecs.T))
    
    # 3. Whiten the Gain Matrix: G_tilde = W * G_proj
    G_tilde = np.dot(W, G_proj)
    
    # --- MNE Scaling Step ---
    n_nzero = np.sum(mask)
    trace_GRGT = np.sum(G_tilde ** 2)
    g_scale = np.sqrt(n_nzero / trace_GRGT)
    print(f"Gain scaling factor: {g_scale:.4e}")
    
    G_tilde = G_tilde * g_scale
    
    # 4. SVD of G_tilde = U * S * V^T
    U, S, Vh = scipy.linalg.svd(G_tilde, full_matrices=False)
    V = Vh.T
    
    # 5. Compute Inverse Operator Weights
    # Gamma = S / (S^2 + lambda2)
    S2 = S ** 2
    Gamma_diag = S / (S2 + lambda2)
    
    # 6. Assemble K
    # K = (V * Gamma) * U^T * W
    KV = V * Gamma_diag  # Broadcast
    KU = np.dot(U.T, W)
    K = np.dot(KV, KU)
    
    # Apply source covariance scaling
    K *= g_scale
    
    # 7. Apply K to data
    print(f"--- [Standalone] Applying {method} Inverse ---")
    source_estimate = np.dot(K, y)
    
    # 8. Apply dSPM normalization if requested
    if method == 'dSPM':
        # noise_var = diag( V * Gamma^2 * V^T ) * g_scale^2
        noise_var = np.sum((V * Gamma_diag) ** 2, axis=1) * (g_scale ** 2)
        noise_norm = 1.0 / np.sqrt(noise_var)
        source_estimate *= noise_norm[:, np.newaxis]
        
    return source_estimate
