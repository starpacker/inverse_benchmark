import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(model, u, v):
    """
    Forward operator: Compute visibilities from sky brightness model.
    V(u,v) = FT{I}(u,v) sampled at (u,v) points.
    
    Parameters
    ----------
    model : 2D array - sky brightness distribution I(l,m)
    u, v : arrays - (u,v) coordinates
    
    Returns
    -------
    visibilities : complex array - predicted visibilities
    valid : boolean array - mask for valid (u,v) points within grid
    """
    n = model.shape[0]
    
    # Compute full FFT of model
    model_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(model)))
    
    # Sample at (u,v) points by nearest-neighbor interpolation
    u_pix = np.round(u + n // 2).astype(int)
    v_pix = np.round(v + n // 2).astype(int)
    
    # Determine valid range
    valid = (u_pix >= 0) & (u_pix < n) & (v_pix >= 0) & (v_pix < n)
    
    n_vis = len(u)
    visibilities = np.zeros(n_vis, dtype=complex)
    
    u_pix_valid = u_pix[valid]
    v_pix_valid = v_pix[valid]
    visibilities[valid] = model_fft[v_pix_valid, u_pix_valid]
    
    return visibilities, valid
