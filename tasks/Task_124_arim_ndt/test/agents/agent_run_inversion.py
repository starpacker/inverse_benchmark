import numpy as np

import matplotlib

matplotlib.use('Agg')

import time

from scipy.signal import hilbert

from scipy.ndimage import gaussian_filter, maximum_filter, label

def run_inversion(data_dict):
    """
    Run TFM (Total Focusing Method) inversion/reconstruction.
    
    This implements delay-and-sum beamforming with analytic signal
    (Hilbert transform) envelope detection.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary from load_and_preprocess_data.
    
    Returns
    -------
    result_dict : dict
        Contains reconstructed image and timing info.
    """
    fmc = data_dict['fmc']
    element_positions = data_dict['element_positions']
    x_grid = data_dict['x_grid']
    z_grid = data_dict['z_grid']
    fs = data_dict['fs']
    c_sound = data_dict['c_sound']
    
    print("\n[2/4] Running TFM reconstruction...")
    t0 = time.time()
    
    n_elem = fmc.shape[0]
    n_samples = fmc.shape[2]
    
    # Compute analytic signal for each TX-RX pair
    print("  Computing analytic signal (Hilbert transform)...")
    fmc_analytic = np.zeros_like(fmc, dtype=np.complex128)
    for i in range(n_elem):
        fmc_analytic[i, :, :] = hilbert(fmc[i, :, :], axis=-1)
    
    # Precompute distances from each element to each pixel
    print("  Precomputing distance tables...")
    ex = element_positions[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    gx = x_grid[np.newaxis, np.newaxis, :]             # (1, 1, NX)
    gz = z_grid[np.newaxis, :, np.newaxis]              # (1, NZ, 1)
    distances = np.sqrt((ex - gx) ** 2 + gz ** 2)      # (N, NZ, NX)
    
    # Convert distances to sample indices (fractional)
    delay_samples = distances / c_sound * fs  # (N, NZ, NX)
    
    image = np.zeros((len(z_grid), len(x_grid)))
    
    print(f"  Delay-and-sum over {n_elem}x{n_elem} element pairs...")
    t_das = time.time()
    for tx in range(n_elem):
        if tx % 8 == 0:
            elapsed = time.time() - t_das
            print(f"    TX {tx}/{n_elem}  ({elapsed:.1f}s elapsed)")
        d_tx = delay_samples[tx]  # (NZ, NX)
        for rx in range(n_elem):
            d_rx = delay_samples[rx]  # (NZ, NX)
            total_delay = d_tx + d_rx  # round-trip delay in samples
            
            # Integer and fractional parts for linear interpolation
            idx = total_delay.astype(np.int64)
            frac = total_delay - idx
            
            # Mask valid indices
            valid = (idx >= 0) & (idx < n_samples - 1)
            
            # Safe index (clamp for out-of-range)
            idx_safe = np.clip(idx, 0, n_samples - 2)
            
            # Linearly interpolate analytic signal
            val = ((1.0 - frac) * fmc_analytic[tx, rx, idx_safe] +
                   frac * fmc_analytic[tx, rx, idx_safe + 1])
            val[~valid] = 0.0
            
            image += np.abs(val)
    
    recon_time = time.time() - t0
    print(f"  TFM reconstruction completed in {recon_time:.1f}s")
    
    # Post-processing
    # Step 1: mild smooth to reduce pixel-level noise
    recon_smooth = gaussian_filter(image, sigma=1.0)
    
    # Step 2: normalize to [0, 1]
    recon_norm = recon_smooth / recon_smooth.max()
    
    # Step 3: suppress background by applying a soft threshold
    bg_threshold = 0.10
    recon_norm = np.where(recon_norm > bg_threshold,
                          (recon_norm - bg_threshold) / (1.0 - bg_threshold),
                          0.0)
    
    # Step 4: power-law compression to sharpen peaks
    recon_norm = recon_norm ** 1.5
    
    # Re-normalize to [0, 1]
    if recon_norm.max() > 0:
        recon_norm = recon_norm / recon_norm.max()
    
    result_dict = {
        'recon_image_raw': image,
        'recon_image': recon_norm,
        'recon_time': recon_time
    }
    
    return result_dict
