import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon

def forward_operator(x, theta):
    """
    Radon transform forward operator: projects a 2D image into a sinogram.

    Parameters
    ----------
    x     : np.ndarray (H, W) image
    theta : np.ndarray projection angles in degrees

    Returns
    -------
    y_pred : np.ndarray sinogram
    """
    y_pred = radon(x, theta=theta)
    return y_pred
