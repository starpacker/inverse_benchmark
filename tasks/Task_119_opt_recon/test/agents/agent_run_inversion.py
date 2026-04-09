import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon

def run_inversion(sinograms_noisy, theta, output_shape):
    """
    Perform Filtered Back-Projection (FBP) reconstruction on noisy sinograms.
    
    Args:
        sinograms_noisy: 3D array of noisy sinograms (nz, sino_height, n_angles)
        theta: projection angles in degrees
        output_shape: tuple (nz, ny, nx) for the output reconstruction
    
    Returns:
        reconstruction: 3D reconstructed volume (nz, ny, nx)
    """
    nz, ny, nx = output_shape
    reconstruction = np.zeros((nz, ny, nx), dtype=np.float64)
    
    for z_idx in range(nz):
        # FBP reconstruction using Ram-Lak (ramp) filter
        recon_slice = iradon(sinograms_noisy[z_idx], theta=theta, circle=True, filter_name='ramp')
        
        # iradon may return different size, crop/pad to match
        rh, rw = recon_slice.shape
        
        # Center-crop or center-pad to (ny, nx)
        sy = max(0, (rh - ny) // 2)
        sx = max(0, (rw - nx) // 2)
        dy = max(0, (ny - rh) // 2)
        dx = max(0, (nx - rw) // 2)
        h_copy = min(rh, ny)
        w_copy = min(rw, nx)
        
        reconstruction[z_idx, dy:dy+h_copy, dx:dx+w_copy] = recon_slice[sy:sy+h_copy, sx:sx+w_copy]
    
    return reconstruction
