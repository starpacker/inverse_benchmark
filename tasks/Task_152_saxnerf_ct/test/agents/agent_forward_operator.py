import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

def forward_operator(x, angles):
    """
    Forward Radon transform operator.
    
    Args:
        x: Input image (size x size)
        angles: Array of projection angles in degrees
        
    Returns:
        y_pred: Sinogram (n_detectors x n_angles)
    """
    return radon(x, theta=angles, circle=True)
