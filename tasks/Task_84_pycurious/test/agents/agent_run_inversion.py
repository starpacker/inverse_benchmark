import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def run_inversion(grid_noisy, xmin, xmax, ymin, ymax, centroids, window_size):
    """
    Inverse: Fit Curie depth parameters from magnetic anomaly grid.
    
    Uses pycurious CurieOptimise class which:
    1. Extracts subgrid around each centroid
    2. Computes radial power spectrum
    3. Fits Bouligand model via scipy.optimize.minimize
    
    Parameters:
        grid_noisy: Input magnetic anomaly grid
        xmin, xmax, ymin, ymax: Grid extent (m)
        centroids: List of (xc, yc) centroid coordinates
        window_size: Window size for spectral analysis (m)
    
    Returns:
        results: List of dictionaries with fitted parameters
        grid_obj: CurieOptimise object for further analysis
    """
    from pycurious import CurieOptimise
    
    # Create CurieOptimise object
    grid_obj = CurieOptimise(grid_noisy, xmin, xmax, ymin, ymax)
    
    results = []
    for i, (xc, yc) in enumerate(centroids):
        print(f"\n  [FIT] Centroid {i+1}: ({xc/1e3:.0f}, {yc/1e3:.0f}) km")
        
        try:
            beta_fit, zt_fit, dz_fit, C_fit = grid_obj.optimise(
                window_size, xc, yc,
                beta=3.0, zt=1.0, dz=20.0, C=5.0,
                taper=np.hanning
            )
            
            curie_depth = zt_fit + dz_fit
            
            result = {
                'xc': xc, 'yc': yc,
                'beta': float(beta_fit),
                'zt': float(zt_fit),
                'dz': float(dz_fit),
                'C': float(C_fit),
                'curie_depth': float(curie_depth),
            }
            
            print(f"  [FIT] Fitted: beta={beta_fit:.2f}, zt={zt_fit:.2f}, "
                  f"dz={dz_fit:.2f}, C={C_fit:.2f}")
            print(f"  [FIT] Curie depth: {curie_depth:.2f} km")
            
        except Exception as e:
            print(f"  [FIT] ERROR: {e}")
            result = {
                'xc': xc, 'yc': yc,
                'beta': np.nan, 'zt': np.nan,
                'dz': np.nan, 'C': np.nan,
                'curie_depth': np.nan,
            }
        
        results.append(result)
    
    return results, grid_obj
