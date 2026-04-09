import matplotlib

matplotlib.use('Agg')

def forward_operator(M_sample, W):
    """
    Apply the DRR forward model: compute intensities from Mueller matrix.
    
    The forward model computes:
        I = W @ vec(M_sample)
    
    where W is the measurement matrix encoding the PSG and PSA configurations.
    
    Parameters
    ----------
    M_sample : ndarray, shape (4, 4)
        Sample Mueller matrix.
    W : ndarray, shape (N, 16)
        Measurement matrix.
    
    Returns
    -------
    I_pred : ndarray, shape (N,)
        Predicted intensity measurements.
    """
    m_vec = M_sample.ravel()
    I_pred = W @ m_vec
    return I_pred
