import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def forward_operator(reflectivity, wavelet):
    """
    Forward operator: Convolve reflectivity with source wavelet.
    s(t) = w(t) * r(t) using pylops Convolve1D operator.
    
    Args:
        reflectivity: reflectivity array (nt,) or (nt, n_traces)
        wavelet: source wavelet array
    
    Returns:
        seismic: convolved seismic data, same shape as reflectivity
    """
    from pylops.signalprocessing import Convolve1D
    
    # Handle both 1D and 2D cases
    if reflectivity.ndim == 1:
        nt = reflectivity.shape[0]
        Cop = Convolve1D(nt, h=wavelet, offset=len(wavelet)//2)
        seismic = Cop @ reflectivity
    else:
        nt, n_traces = reflectivity.shape
        seismic = np.zeros_like(reflectivity)
        
        for j in range(n_traces):
            # Create convolution operator
            Cop = Convolve1D(nt, h=wavelet, offset=len(wavelet)//2)
            seismic[:, j] = Cop @ reflectivity[:, j]
    
    return seismic
