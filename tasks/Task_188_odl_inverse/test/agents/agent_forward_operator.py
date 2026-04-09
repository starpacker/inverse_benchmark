import warnings

import matplotlib

matplotlib.use('Agg')

import numpy as np

warnings.filterwarnings('ignore')

def forward_operator(x, ray_transform):
    """
    Apply the forward operator (ray transform / projection) to an image.
    
    This implements the CT forward model: y = Ax, where A is the ray transform
    that computes line integrals through the image along each projection angle.
    
    Parameters
    ----------
    x : numpy.ndarray or odl.DiscretizedSpaceElement
        Input image to project.
    ray_transform : odl.tomo.RayTransform
        The ray transform operator.
    
    Returns
    -------
    numpy.ndarray
        The computed sinogram (projection data).
    """
    # If input is a numpy array, convert to ODL element
    if isinstance(x, np.ndarray):
        x_element = ray_transform.domain.element(x)
    else:
        x_element = x
    
    # Apply ray transform
    y_pred = ray_transform(x_element)
    
    return y_pred.asarray()
