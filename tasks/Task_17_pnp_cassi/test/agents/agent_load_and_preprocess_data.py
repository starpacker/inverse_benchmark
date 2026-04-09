import numpy as np
import scipy.io as sio

def load_and_preprocess_data(matfile, maskfile, step):
    """
    Load and preprocess data for CASSI reconstruction.
    Returns truth, mask_3d, and simulated measurement.
    
    Parameters:
    - matfile: str, path to the .mat file containing the ground truth image (key 'img')
    - maskfile: str, path to the .mat file containing the mask (key 'mask')
    - step: int, the number of pixels to shift per spectral channel
    
    Returns:
    - truth: np.ndarray, the normalized ground truth spectral cube
    - mask_3d: np.ndarray, the 3D mask volume accounting for dispersion
    - meas: np.ndarray, the simulated 2D compressive measurement
    """
    # 1. Load Truth
    truth = sio.loadmat(matfile)['img']
    
    # Normalize truth to [0, 1] if not already
    if truth.max() > 1:
        truth = truth / 255.0
        
    r, c, nC = truth.shape
    
    # 2. Load and Prepare Mask
    mask256 = sio.loadmat(maskfile)['mask']
    
    # Calculate the width of the dispersed image
    dispersed_width = c + step * (nC - 1)
    
    # Initialize the 3D mask volume
    mask_3d = np.zeros((r, dispersed_width, nC))
    
    # Place the mask into the 3D volume, shifting it for each channel
    for i in range(nC):
        mask_3d[:, i*step : i*step + c, i] = mask256
    
    # 3. Shift truth for measurement simulation
    truth_shift = np.zeros((r, dispersed_width, nC))
    
    for i in range(nC):
        truth_shift[:, i * step : i * step + c, i] = truth[:, :, i]
    
    # 4. Generate Measurement (Simulation)
    meas = np.sum(mask_3d * truth_shift, axis=2)
    
    return truth, mask_3d, meas