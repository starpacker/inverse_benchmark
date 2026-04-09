import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def add_noise(data, noise_level):
    """Add complex Gaussian noise (relative to amplitude)."""
    amplitude = np.abs(data)
    noise_real = noise_level * amplitude * np.random.randn(*data.shape)
    noise_imag = noise_level * amplitude * np.random.randn(*data.shape)
    return data + noise_real + 1j * noise_imag

def load_and_preprocess_data(frequencies, offsets, water_depth, src_depth, rec_depth, noise_level, seed=42):
    """
    Create synthetic CSEM data by:
    1. Defining a true 1D marine resistivity model
    2. Computing the EM response using empymod
    3. Adding noise to create observed data
    
    Returns:
        data_dict: Dictionary containing all necessary data and parameters
    """
    np.random.seed(seed)
    
    # Create true 1D marine CSEM model
    depth = [0, water_depth]  # Air-sea, sea-seafloor
    depth += [water_depth + d for d in [200, 500]]  # Two subsurface interfaces
    
    # Resistivities: [air, seawater, overburden, reservoir, basement]
    res_true = [
        2e14,    # Air
        0.3,     # Seawater
        1.0,     # Overburden sediment (0-200m sub-seafloor)
        30.0,    # Reservoir (200-500m) — TARGET
        1.5,     # Basement (below 500m)
    ]
    
    n_layers_subsurface = 3  # overburden + reservoir + basement
    
    # Indices of subsurface parameters to invert (skip air + seawater)
    param_indices = list(range(2, len(res_true)))
    
    # Source and receiver absolute depths
    src_z = water_depth + src_depth
    rec_z = water_depth + rec_depth
    
    # Compute true EM response
    data_true = forward_operator(
        depth=depth,
        res=res_true,
        frequencies=frequencies,
        offsets=offsets,
        src_depth_abs=src_z,
        rec_depth_abs=rec_z
    )
    
    # Add noise to create observed data
    data_obs = add_noise(data_true, noise_level)
    
    # Initial model (perturbed from true)
    res_init = list(res_true[:2])  # Keep air + seawater fixed
    res_init += [2.0, 5.0, 2.0]  # 3 subsurface layers (reasonable guesses)
    
    data_dict = {
        'depth': depth,
        'res_true': res_true,
        'res_init': res_init,
        'n_layers_subsurface': n_layers_subsurface,
        'param_indices': param_indices,
        'frequencies': frequencies,
        'offsets': offsets,
        'src_z': src_z,
        'rec_z': rec_z,
        'data_true': data_true,
        'data_obs': data_obs,
        'noise_level': noise_level,
        'water_depth': water_depth,
    }
    
    return data_dict

def forward_operator(depth, res, frequencies, offsets, src_depth_abs, rec_depth_abs):
    """
    Compute CSEM frequency-domain EM response using empymod.
    
    Args:
        depth: List of layer interface depths
        res: List of layer resistivities
        frequencies: Array of frequencies (Hz)
        offsets: Array of source-receiver offsets (m)
        src_depth_abs: Absolute source depth (m)
        rec_depth_abs: Absolute receiver depth (m)
    
    Returns:
        response: Complex E-field array of shape (n_freq, n_off)
    """
    import empymod
    
    n_freq = len(frequencies)
    n_off = len(offsets)
    response = np.zeros((n_freq, n_off), dtype=complex)
    
    for j, offset in enumerate(offsets):
        resp = empymod.dipole(
            src=[0, 0, src_depth_abs],
            rec=[offset, 0, rec_depth_abs],
            depth=depth,
            res=res,
            freqtime=frequencies,
            verb=0,
        )
        response[:, j] = resp
    
    return response
