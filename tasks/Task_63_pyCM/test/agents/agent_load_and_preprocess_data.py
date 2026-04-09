import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter

def load_and_preprocess_data(nx, ny, dx, dy, E, nu, noise_level, seed):
    """
    Generate synthetic contour method data including:
    - Building the influence matrix (compliance matrix)
    - Creating ground truth stress field
    - Computing surface displacements with noise
    
    Parameters
    ----------
    nx, ny : int       Grid dimensions.
    dx, dy : float     Grid spacing [mm].
    E : float          Young's modulus [MPa].
    nu : float         Poisson's ratio.
    noise_level : float  Noise level for displacement [mm].
    seed : int         Random seed.
    
    Returns
    -------
    C : ndarray (nx*ny, nx*ny)   Influence matrix.
    stress_gt_vec : ndarray (nx*ny,)  Ground truth stress vector [MPa].
    stress_gt_2d : ndarray (nx, ny)   Ground truth stress 2D [MPa].
    disp_clean : ndarray (nx*ny,)     Clean displacement [mm].
    disp_noisy : ndarray (nx*ny,)     Noisy displacement [mm].
    """
    print("[DATA] Building influence matrix ...")
    n = nx * ny
    C = np.zeros((n, n))
    
    # Grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Boussinesq: displacement due to point load on half-space
    # u_z(r) = (1-ν²)/(πE) · P/r  for normal load P at surface
    prefactor = (1 - nu**2) / (np.pi * E)
    
    for j in range(n):
        # Distance from source j to all observation points
        r = np.sqrt((coords[:, 0] - coords[j, 0])**2 +
                    (coords[:, 1] - coords[j, 1])**2)
        r = np.maximum(r, dx * 0.1)  # regularise singularity
        
        # Compliance: displacement per unit stress × patch area
        C[:, j] = prefactor * dx * dy / r
    
    print("[DATA] Creating ground truth stress ...")
    # Create realistic residual stress pattern
    x_norm = np.linspace(-1, 1, nx)
    y_norm = np.linspace(0, 1, ny)
    xx_norm, yy_norm = np.meshgrid(x_norm, y_norm, indexing='ij')
    
    # Parabolic profile (typical weld residual stress)
    stress_gt_2d = 200 * (1 - 2 * xx_norm**2) * np.exp(-3 * yy_norm)
    
    # Add localised feature
    r_loc = np.sqrt((xx_norm - 0.3)**2 + (yy_norm - 0.3)**2)
    stress_gt_2d += 150 * np.exp(-r_loc**2 / 0.05)
    
    # Smooth
    stress_gt_2d = gaussian_filter(stress_gt_2d, sigma=1.0)
    stress_gt_vec = stress_gt_2d.ravel()
    
    print("[DATA] Computing surface displacements ...")
    disp_clean = C @ stress_gt_vec
    
    rng = np.random.default_rng(seed)
    disp_noisy = disp_clean + noise_level * rng.standard_normal(len(disp_clean))
    
    print(f"[DATA] Stress range: [{stress_gt_vec.min():.1f}, {stress_gt_vec.max():.1f}] MPa")
    print(f"[DATA] Displacement range: [{disp_clean.min():.4f}, {disp_clean.max():.4f}] mm")
    
    return C, stress_gt_vec, stress_gt_2d, disp_clean, disp_noisy
