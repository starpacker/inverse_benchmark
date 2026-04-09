import numpy as np

def forward_operator(x, Phi):
    """
    Forward model of snapshot compressive imaging (SCI).
    Multiple encoded frames are collapsed into a single measurement.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The 3D scene datacube with shape (Height, Width, B), where B is 
        frames or spectral bands.
    Phi : numpy.ndarray
        The 3D sensing matrix (Mask) with shape (Height, Width, B).
        
    Returns:
    --------
    numpy.ndarray
        The 2D compressed measurement with shape (Height, Width).
    """
    # Element-wise multiplication followed by summation along the 3rd dimension (axis 2)
    return np.sum(x * Phi, axis=2)