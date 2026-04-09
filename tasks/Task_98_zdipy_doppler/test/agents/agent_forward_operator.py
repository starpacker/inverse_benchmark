import matplotlib

matplotlib.use("Agg")

def forward_operator(A, B):
    """
    Compute observed line profiles d = A @ B.

    Parameters
    ----------
    A : ndarray of shape (n_phases * n_vbins, n_pix)
        Design matrix
    B : ndarray of shape (n_pix,)
        Surface brightness map

    Returns
    -------
    d : ndarray of shape (n_phases * n_vbins,)
        Predicted line profile data
    """
    return A @ B
