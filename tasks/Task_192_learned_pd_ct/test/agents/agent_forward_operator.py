import matplotlib

matplotlib.use('Agg')

from skimage.transform import resize, radon, iradon

def forward_operator(x, theta_angles):
    """
    Forward operator: Radon transform (parallel beam CT).
    
    Parameters:
    -----------
    x : ndarray
        2D image to transform
    theta_angles : ndarray
        Projection angles in degrees
    
    Returns:
    --------
    y_pred : ndarray
        Sinogram (Radon transform of x)
    """
    y_pred = radon(x, theta=theta_angles, circle=False)
    return y_pred
