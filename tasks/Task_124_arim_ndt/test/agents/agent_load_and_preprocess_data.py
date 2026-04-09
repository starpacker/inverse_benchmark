import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import time

def generate_toneburst(fc, n_cycles, fs):
    """Generate a Hanning-windowed tone burst."""
    duration = n_cycles / fc
    n_pts = int(np.ceil(duration * fs))
    t_burst = np.arange(n_pts) / fs
    burst = np.sin(2 * np.pi * fc * t_burst) * np.hanning(n_pts)
    return t_burst, burst

def compute_element_positions(n_elements, pitch):
    """Return 1-D array of element x-positions centered on 0."""
    return (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch

def load_and_preprocess_data(
    defects_mm,
    n_elements,
    pitch,
    freq,
    c_sound,
    bandwidth,
    fs,
    n_samples,
    snr_db,
    n_cycles,
    x_min,
    x_max,
    z_min,
    z_max,
    nx,
    nz,
    results_dir
):
    """
    Load and preprocess data for TFM imaging.
    
    This function:
    1. Sets up array geometry
    2. Generates toneburst pulse
    3. Synthesizes Full Matrix Capture (FMC) data from defects
    4. Creates imaging grid
    5. Creates ground truth defect map
    
    Returns
    -------
    data_dict : dict
        Contains all preprocessed data needed for forward operator and inversion.
    """
    np.random.seed(42)
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert defect positions to metres
    defects_m = [(x * 1e-3, z * 1e-3) for x, z in defects_mm]
    
    # Element positions
    element_positions = compute_element_positions(n_elements, pitch)
    print(f"Array: {n_elements} elements, pitch={pitch*1e3:.2f} mm, "
          f"aperture={element_positions[-1]-element_positions[0]:.1f} mm")
    
    # Generate toneburst
    _, toneburst = generate_toneburst(freq, n_cycles, fs)
    print(f"Toneburst: {freq/1e6:.1f} MHz, {n_cycles} cycles, "
          f"{len(toneburst)} samples")
    
    # Synthesize FMC data
    print("\n[1/4] Synthesizing Full Matrix Capture data...")
    t0 = time.time()
    
    n_elem = len(element_positions)
    fmc = np.zeros((n_elem, n_elem, n_samples))
    t_axis = np.arange(n_samples) / fs
    burst_len = len(toneburst)
    
    for dx, dz in defects_m:
        # Distances from each element to this defect
        dist = np.sqrt((element_positions - dx) ** 2 + dz ** 2)
        
        for tx in range(n_elem):
            d_tx = dist[tx]
            for rx in range(n_elem):
                d_rx = dist[rx]
                delay = (d_tx + d_rx) / c_sound
                amplitude = 1.0 / (d_tx * d_rx) * 1e-3  # geometric spreading
                sample_start = int(round(delay * fs))
                sample_end = sample_start + burst_len
                if sample_end < n_samples:
                    fmc[tx, rx, sample_start:sample_end] += amplitude * toneburst
    
    # Add white Gaussian noise
    signal_power = np.mean(fmc ** 2)
    if signal_power > 0:
        noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
        fmc += np.random.randn(*fmc.shape) * noise_std
    
    print(f"  FMC shape: {fmc.shape}, generated in {time.time()-t0:.1f}s")
    
    # Create imaging grid
    x_grid = np.linspace(x_min, x_max, nx)
    z_grid = np.linspace(z_min, z_max, nz)
    print(f"\nImaging grid: {nx} x {nz} = {nx*nz} pixels")
    print(f"  x: [{x_min*1e3:.1f}, {x_max*1e3:.1f}] mm")
    print(f"  z: [{z_min*1e3:.1f}, {z_max*1e3:.1f}] mm")
    
    # Create ground truth map
    spot_sigma = 0.8e-3
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    gt_map = np.zeros_like(XX)
    for dx, dz in defects_m:
        gt_map += np.exp(-((XX - dx) ** 2 + (ZZ - dz) ** 2) / (2 * spot_sigma ** 2))
    gt_map /= gt_map.max()
    
    # Save FMC data
    np.save(os.path.join(results_dir, "fmc_data.npy"), fmc)
    
    data_dict = {
        'fmc': fmc,
        't_axis': t_axis,
        'element_positions': element_positions,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'gt_map': gt_map,
        'defects_m': defects_m,
        'defects_mm': defects_mm,
        'fs': fs,
        'c_sound': c_sound,
        'n_elements': n_elements,
        'pitch': pitch,
        'freq': freq,
        'snr_db': snr_db,
        'nx': nx,
        'nz': nz,
        'results_dir': results_dir
    }
    
    return data_dict
