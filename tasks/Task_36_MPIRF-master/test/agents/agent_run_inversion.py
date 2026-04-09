import numpy as np

def run_inversion(system_matrix, measurements, config_data, iterations=20, lambd=1e-6):
    """
    Reconstructs the image using the Kaczmarz algorithm (Algebraic Reconstruction Technique).
    Solves Ax = b for x.
    """
    
    A = system_matrix
    b = measurements
    
    M = A.shape[0] # Number of equations (frequency components)
    N = A.shape[1] # Number of unknowns (pixels)
    
    x = np.zeros(N, dtype=b.dtype)
    residual = np.zeros(M, dtype=x.dtype)
    
    # Precompute row energy
    energy = np.zeros(M, dtype=np.double)
    for m in range(M):
        energy[m] = np.linalg.norm(A[m, :])
        
    row_index_cycle = np.arange(0, M)
    
    # Kaczmarz Iterations
    for l in range(iterations):
        for m in range(M):
            k = row_index_cycle[m]
            if energy[k] > 0:
                # beta = (b[k] - <a_k, x> - sqrt(lambda)*residual[k]) / (||a_k||^2 + lambda)
                dot_prod = A[k, :].dot(x)
                numerator = b[k] - dot_prod - np.sqrt(lambd) * residual[k]
                denominator = (energy[k] ** 2 + lambd)
                
                beta = numerator / denominator
                
                # Update x
                x[:] += beta * A[k, :].conjugate()
                
                # Update residual
                residual[k] += np.sqrt(lambd) * beta

    # --- Reshape and Post-process Result ---
    xn = config_data['xn']
    yn = config_data['yn']
    
    # The result x is complex, but the physical distribution is real. 
    # Take real part and reshape.
    c_recon = np.real(np.reshape(x, (xn, yn)))
    
    # Crop borders (artifact removal standard in this legacy code)
    c_recon_cropped = c_recon[1:-1, 1:-1]
    
    # Normalize
    max_val = np.max(c_recon_cropped)
    if max_val > 0:
        c_recon_norm = c_recon_cropped / max_val
    else:
        c_recon_norm = c_recon_cropped
        
    return c_recon_norm
