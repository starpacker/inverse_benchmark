import numpy as np
from tqdm import tqdm

def run_inversion(processed_data, geometry, fs, speed_of_sound, n_pixels, field_of_view):
    """
    Run the inversion (reconstruction) using Delay-and-Sum Backprojection.
    
    Args:
        processed_data: Preprocessed signal data, shape (n_wl, n_det, n_time)
        geometry: Detector positions, shape (n_det, 3)
        fs: Sampling frequency
        speed_of_sound: Speed of sound
        n_pixels: Tuple (nx, ny, nz) for reconstruction grid
        field_of_view: Tuple (lx, ly, lz) for physical dimensions
        
    Returns:
        dict containing reconstruction
    """
    signal = processed_data
    
    nx, ny, nz = n_pixels
    lx, ly, lz = field_of_view
    
    # Create coordinate vectors centered around 0
    xs = np.linspace(-lx/2, lx/2, nx)
    ys = np.linspace(-ly/2, ly/2, ny)
    zs = np.array([0.0]) # Currently reconstructing a z=0 slice
    
    # Create 3D grid
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    
    n_wl = signal.shape[0]
    n_det = signal.shape[1]
    
    # Initialize output array
    reconstruction = np.zeros((n_wl, nz, ny, nx))
    
    # Calculate spatial resolution (distance per time sample)
    dl = speed_of_sound / fs
    
    print(f"Reconstructing {n_wl} wavelengths...")
    
    for i_wl in range(n_wl):
        print(f"  Wavelength {i_wl+1}/{n_wl}...")
        sig_wl = signal[i_wl]
        
        # Iterate over all detectors
        for i_det in tqdm(range(n_det), leave=False):
            det_pos = geometry[i_det]
            
            # 1. Calculate distance from this detector to all pixels
            dist = np.sqrt((X - det_pos[0])**2 + (Y - det_pos[1])**2 + (Z - det_pos[2])**2)
            
            # 2. Convert distance to time sample index
            sample_idx = (dist / dl).astype(int)
            
            # 3. Check bounds (ignore pixels too close or too far for the time window)
            valid_mask = (sample_idx >= 0) & (sample_idx < sig_wl.shape[-1])
            
            # 4. Sum the signal value at the calculated time index into the pixel
            # Note: We use fancy indexing here. 
            # sig_wl[i_det, sample_idx[valid_mask]] grabs the specific samples needed.
            reconstruction[i_wl][valid_mask] += sig_wl[i_det, sample_idx[valid_mask]]
    
    return {
        'reconstruction': reconstruction
    }