import numpy as np


# --- Extracted Dependencies ---

def zernike_mode_explicit(n, m, X, Y, D):
    """
    Generates a Zernike mode Z_n^m on the grid (X, Y)
    """
    # Normalized coordinates
    R = np.sqrt(X**2 + Y**2) / (D / 2)
    Theta = np.arctan2(Y, X)
    
    # Mask outside pupil
    mask = R <= 1.0
    
    # Initialize Z
    Z = np.zeros_like(X)
    
    # Calculate R_nm only inside pupil
    R_vals = R[mask]
    Theta_vals = Theta[mask]
    
    # Radial function values
    Rad = np.zeros_like(R_vals)
    if (n - m) % 2 == 0:
        for k in range((n - m) // 2 + 1):
            if (n - k) < 0 or ((n + m) // 2 - k) < 0 or ((n - m) // 2 - k) < 0:
                continue
            num = ((-1)**k) * np.math.factorial(n - k)
            denom = (np.math.factorial(k) * 
                     np.math.factorial((n + m) // 2 - k) * 
                     np.math.factorial((n - m) // 2 - k))
            Rad += (num / denom) * (R_vals**(n - 2 * k))
            
    # Azimuthal part
    if m == 0:
        Z[mask] = np.sqrt(n + 1) * Rad
    elif m > 0:
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.cos(m * Theta_vals)
    else:  # m < 0
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.sin(-m * Theta_vals)
        
    return Z
