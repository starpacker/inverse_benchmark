import warnings

import numpy as np

import matplotlib

matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from cuqi.testproblem import Deconvolution1D

def load_and_preprocess_data(dim, phantom, noise_std, seed=42):
    """
    Load and preprocess data for 1D deconvolution.
    
    Creates a Deconvolution1D test problem and extracts:
    - noisy observed data
    - ground truth signal
    - forward model (convolution operator)
    
    Parameters
    ----------
    dim : int
        Dimension of the 1D signal.
    phantom : str
        Type of phantom signal (e.g., 'sinc').
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'y_data': noisy observed data (numpy array)
        - 'x_true': ground truth signal (numpy array)
        - 'forward_model': linear forward convolution model
        - 'dim': dimension
        - 'noise_std': noise standard deviation
        - 'phantom': phantom type
    """
    np.random.seed(seed)
    
    # Create test problem
    tp = Deconvolution1D(dim=dim, phantom=phantom, noise_std=noise_std)
    
    y_data = np.asarray(tp.data)
    x_true = np.asarray(tp.exactSolution)
    A = tp.model
    
    print(f"Test problem: Deconvolution1D, dim={dim}, phantom='{phantom}'")
    print(f"  Data shape: {y_data.shape}")
    print(f"  Ground truth range: [{x_true.min():.4f}, {x_true.max():.4f}]")
    print(f"  Observation range:  [{y_data.min():.4f}, {y_data.max():.4f}]")
    
    return {
        'y_data': y_data,
        'x_true': x_true,
        'forward_model': A,
        'dim': dim,
        'noise_std': noise_std,
        'phantom': phantom
    }
