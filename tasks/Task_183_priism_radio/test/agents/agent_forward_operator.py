import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(image, ui, vi):
    """
    Forward model: image → visibilities at (u,v) sample points.
    Uses FFT + sampling.
    
    Parameters:
        image: 2D numpy array (ny, nx) - the sky image
        ui: 1D int array - u grid indices
        vi: 1D int array - v grid indices
    
    Returns:
        vis: 1D complex array - sampled visibilities
    """
    ft = np.fft.fft2(image)
    vis = ft[vi, ui]
    return vis
