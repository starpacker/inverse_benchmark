import matplotlib

matplotlib.use("Agg")

def forward_operator(x, geometry_matrix):
    """
    Apply the forward operator (line integral projection) to the emissivity field.
    
    The forward model computes line integrals: y_i = integral_{LOS_i} epsilon(R, Z) dl
    
    Parameters
    ----------
    x : ndarray
        Input emissivity field, either 2D (NR, NZ) or 1D (NR*NZ,)
    geometry_matrix : sparse matrix
        The geometry matrix L containing line-pixel intersection lengths
        
    Returns
    -------
    y_pred : ndarray
        Predicted line-integrated measurements (n_los,)
    """
    # Flatten if 2D
    if x.ndim == 2:
        x_flat = x.ravel()
    else:
        x_flat = x
    
    # Apply geometry matrix (line integrals)
    y_pred = geometry_matrix @ x_flat
    
    return y_pred
