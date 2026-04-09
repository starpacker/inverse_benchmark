import matplotlib

matplotlib.use("Agg")

from pydmd import DMD

def run_inversion(noisy_field, svd_rank):
    """
    Run DMD inversion to recover spatial modes and temporal dynamics.
    
    Standard DMD with truncated SVD decomposes snapshot matrix X ≈ Φ Λ^k B
    recovering spatial modes Φ and discrete eigenvalues μ = exp((σ + jω)·dt).
    
    Parameters
    ----------
    noisy_field : ndarray (n_spatial, nt)
        Noisy snapshot matrix
    svd_rank : int
        Truncation rank for SVD
    
    Returns
    -------
    result : dict
        Contains:
        - 'dmd_object': the fitted DMD object
        - 'reconstruction': reconstructed data (real part)
        - 'modes': spatial DMD modes
        - 'eigenvalues': discrete eigenvalues
        - 'dynamics': temporal dynamics matrix
    """
    dmd = DMD(svd_rank=svd_rank)
    dmd.fit(noisy_field)
    
    reconstruction = dmd.reconstructed_data.real
    modes = dmd.modes
    eigenvalues = dmd.eigs
    dynamics = dmd.dynamics
    
    result = {
        'dmd_object': dmd,
        'reconstruction': reconstruction,
        'modes': modes,
        'eigenvalues': eigenvalues,
        'dynamics': dynamics,
    }
    
    return result
