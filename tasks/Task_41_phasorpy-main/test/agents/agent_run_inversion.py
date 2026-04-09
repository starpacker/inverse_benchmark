import numbers

import numpy as np

from scipy.ndimage import median_filter

def parse_harmonic(harmonic, harmonic_max=None):
    """Parses harmonic input into a list of integers."""
    if harmonic_max is not None and harmonic_max < 1:
        raise ValueError(f'{harmonic_max=} < 1')

    if harmonic is None:
        return [1], False

    if isinstance(harmonic, (int, numbers.Integral)):
        if harmonic < 1 or (harmonic_max is not None and harmonic > harmonic_max):
            raise IndexError(f'{harmonic=!r} is out of bounds [1, {harmonic_max}]')
        return [int(harmonic)], False

    if isinstance(harmonic, str):
        if harmonic == 'all':
            if harmonic_max is None:
                raise TypeError(f'maximum harmonic must be specified for {harmonic=!r}')
            return list(range(1, harmonic_max + 1)), True
        raise ValueError(f'invalid {harmonic=!r}')

    h = np.atleast_1d(harmonic)
    if h.size == 0:
        raise ValueError(f'{harmonic=!r} is empty')
    return [int(i) for i in harmonic], True

def phasor_filter_median_impl(mean, real, imag, size=3, repeat=1):
    """Median filter implementation for phasor coordinates."""
    mean = np.asarray(mean)
    real = np.asarray(real)
    imag = np.asarray(imag)
    
    if repeat == 0:
        return mean, real, imag
        
    for _ in range(repeat):
        real = median_filter(real, size=size)
        imag = median_filter(imag, size=size)
        
    return mean, real, imag

def phasor_threshold_impl(mean, real, imag, mean_min=0):
    """Thresholds phasor coordinates based on intensity."""
    mask = mean < mean_min
    
    mean_out = mean.copy()
    real_out = real.copy()
    imag_out = imag.copy()
    
    mean_out[mask] = np.nan
    real_out[mask] = np.nan
    imag_out[mask] = np.nan
    
    return mean_out, real_out, imag_out

def run_inversion(mean, real, imag, original_axis=0, original_samples=None, harmonic=None, filter_params=None):
    """
    Performs the Inverse Model: Phasor -> Signal (Reconstruction).
    Includes optional filtering step on the phasor coordinates before inversion.
    """
    # 1. Apply Filtering (if params provided)
    if filter_params:
        mean_f, real_f, imag_f = phasor_filter_median_impl(
            mean, real, imag, 
            size=filter_params.get('median_size', 3), 
            repeat=filter_params.get('median_repeat', 1)
        )
        mean_f, real_f, imag_f = phasor_threshold_impl(
            mean_f, real_f, imag_f, 
            mean_min=filter_params.get('threshold_min', 0)
        )
    else:
        mean_f, real_f, imag_f = mean, real, imag

    # 2. Prepare Reconstruction parameters
    mean_arr = np.asarray(mean_f)
    real_arr = np.asarray(real_f)
    imag_arr = np.asarray(imag_f)
    
    harmonic_list, _ = parse_harmonic(harmonic)
    
    if original_samples is None:
        # Default to Nyquist based on max harmonic if unknown
        original_samples = max(harmonic_list) * 2 + 1
        
    phase_grid = np.linspace(0, 2*np.pi, original_samples, endpoint=False)
    
    # 3. Reconstruction Logic: Signal = Mean * (1 + 2 * Sum(G*cos + S*sin))
    
    # Initialize signal with DC component
    # We need to broadcast mean_arr to (samples, ...)
    # Assume mean_arr shape is (Y, X) or similar spatial dims
    
    # Create broadcast shape
    # If mean is (Y, X), result is (samples, Y, X) initially
    target_shape = (original_samples,) + mean_arr.shape
    rec_signal = np.ones(target_shape) * mean_arr[np.newaxis, ...]
    
    # Iterate through harmonics to add modulation
    # Check if input arrays have a harmonic axis
    has_harmonic_dim = (real_arr.ndim == mean_arr.ndim + 1)
    
    if has_harmonic_dim:
        # real_arr is (H, Y, X)
        for h_idx, h_val in enumerate(harmonic_list):
            r = real_arr[h_idx]
            i = imag_arr[h_idx]
            
            # Angle: (samples,)
            angle = phase_grid * h_val
            
            # Reshape angle for broadcasting: (samples, 1, 1)
            angle_reshaped = angle.reshape((-1,) + (1,) * (mean_arr.ndim))
            
            # r, i are (Y, X), broadcast to (1, Y, X)
            r_b = r[np.newaxis, ...]
            i_b = i[np.newaxis, ...]
            
            term = r_b * np.cos(angle_reshaped) + i_b * np.sin(angle_reshaped)
            rec_signal += 2 * mean_arr[np.newaxis, ...] * term
            
    else:
        # Scalar harmonic input
        h_val = harmonic_list[0]
        r = real_arr
        i = imag_arr
        
        angle = phase_grid * h_val
        angle_reshaped = angle.reshape((-1,) + (1,) * (mean_arr.ndim))
        
        r_b = r[np.newaxis, ...]
        i_b = i[np.newaxis, ...]
        
        term = r_b * np.cos(angle_reshaped) + i_b * np.sin(angle_reshaped)
        rec_signal += 2 * mean_arr[np.newaxis, ...] * term

    # 4. Move axis to original position if necessary
    if original_axis != 0:
        rec_signal = np.moveaxis(rec_signal, 0, original_axis)
        
    return rec_signal
