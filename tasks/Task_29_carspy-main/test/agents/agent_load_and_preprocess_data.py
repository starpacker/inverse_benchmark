import numpy as np

def load_and_preprocess_data(raw_signal, nu_axis, noise_level=0.0):
    """
    Loads raw spectral data (simulated here) and performs normalization/preprocessing.
    
    Args:
        raw_signal (np.ndarray): The measured intensity array.
        nu_axis (np.ndarray): The wavenumber axis.
        noise_level (float): Std dev of Gaussian noise to add (for simulation).
    
    Returns:
        tuple: (processed_signal, nu_axis)
    """
    signal = np.array(raw_signal, dtype=float)
    
    # 1. Background subtraction (implied minimal background here)
    bg = 0.0
    signal = signal - bg
    signal[signal < 0] = 0
    
    # 2. Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, signal.shape)
        signal = signal + noise
        signal[signal < 0] = 0
        
    # 3. Normalize
    mx = signal.max()
    if mx > 0:
        signal /= mx
        
    # 4. Square root scaling (often used in CARS to work with Chi, not |Chi|^2, but fitting usually on Intensity)
    # Keeping it as Intensity (I ~ |Chi|^2) for this standard implementation
    
    return signal, nu_axis
