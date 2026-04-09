import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon

def forward_operator(phantom_slice, theta):
    """
    Compute the Radon transform (sinogram) of a 2D slice.
    This simulates parallel-beam optical projections at given angles.
    
    Args:
        phantom_slice: 2D array representing one slice of the phantom
        theta: array of projection angles in degrees
    
    Returns:
        sinogram: 2D array where each column is a projection at the corresponding angle
    """
    sinogram = radon(phantom_slice, theta=theta, circle=True)
    return sinogram
