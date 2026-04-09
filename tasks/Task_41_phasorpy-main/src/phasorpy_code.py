import os
import sys
import math
import logging
import numbers
import warnings
import numpy as np
import scipy.ndimage
from scipy.ndimage import median_filter
import tifffile
import xarray
import pooch

# =============================================================================
# Utils and Helper Functions (Must be defined first)
# =============================================================================

def parse_signal_axis(signal, axis=None):
    """Identifies the signal axis for processing."""
    if hasattr(signal, 'dims'):
        if axis is None:
            for ax in 'HC':
                if ax in signal.dims:
                    return signal.dims.index(ax)
            return -1
        if isinstance(axis, int):
            return axis
        if axis in signal.dims:
            return signal.dims.index(axis)
        raise ValueError(f'{axis=} not found in {signal.dims!r}')
    if axis is None:
        return -1
    if isinstance(axis, int):
        return axis
    raise ValueError(f'invalid {axis=} for {type(signal)=}')

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

def fetch(fname):
    """Fetches data from remote or local source."""
    if os.path.exists(fname):
        return os.path.abspath(fname)
        
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    file_path = os.path.join(script_dir, fname)
    if os.path.exists(file_path):
        return file_path

    if fname == 'Embryo.tif':
        url = 'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_8046636/Embryo.tif'
        file_hash = 'd1107de8d0f3da476e90bcb80ddf40231df343ed9f28340c873cf858ca869e20'
        return pooch.retrieve(
            url=url,
            known_hash='sha256:' + file_hash,
            fname=fname,
            path=pooch.os_cache('phasorpy'),
            progressbar=True
        )
    return fname

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

# =============================================================================
# Core Functional Components
# =============================================================================

def load_and_preprocess_data(filename, harmonic='all'):
    """
    Loads image data and determines dimensions.
    Returns the raw data array and preprocessing metadata.
    """
    local_path = fetch(filename)
    
    # Load using tifffile
    with tifffile.TiffFile(local_path) as tif:
        data = tif.asarray()
        
        # Heuristic for Embryo.tif or general timelapse FLIM data
        # Embryo.tif is typically (Time, Y, X) = (56, 128, 128)
        shape = data.shape
        dims = list('CYX') # Default guess
        
        if len(shape) == 3:
            # Assume (Time/Harmonic, Y, X)
            dims = ['H', 'Y', 'X'] 
        elif len(shape) == 4:
            dims = ['C', 'H', 'Y', 'X']
            
        # Basic normalization or type conversion if needed
        data = data.astype(np.float32)

    return data, {'dims': dims, 'shape': shape, 'source': local_path}


def forward_operator(signal, axis=0, harmonic=1):
    """
    Computes the Forward Model: Signal (Time Domain) -> Phasor (Frequency Domain).
    Returns (mean, real, imag) components.
    """
    signal = np.asarray(signal)
    
    # Handle negative indexing or dimension matching
    ndim = signal.ndim
    if axis < 0:
        axis += ndim
        
    samples = signal.shape[axis]
    harmonic_list, has_harmonic_axis = parse_harmonic(harmonic, samples // 2)
    
    # Prepare for FFT: move signal axis to last
    if axis != ndim - 1:
        signal_swapped = np.swapaxes(signal, axis, -1)
    else:
        signal_swapped = signal

    # Real FFT
    # signal_swapped shape: (..., samples)
    fft_values = np.fft.rfft(signal_swapped, axis=-1)
    
    # Extract DC (0th component)
    dc = fft_values[..., 0].real
    
    # Avoid division by zero
    valid_dc = np.abs(dc) > 1e-9
    
    means = dc / samples
    
    reals = []
    imags = []
    
    for h in harmonic_list:
        if h < fft_values.shape[-1]:
            val = fft_values[..., h]
            
            # Normalization logic:
            # Phasor G (real) = Re(DFT) / DC
            # Phasor S (imag) = -Im(DFT) / DC  (Note the sign flip for standard phasor definition)
            
            r = np.zeros_like(dc)
            i = np.zeros_like(dc)
            
            r[valid_dc] = val.real[valid_dc] / dc[valid_dc]
            i[valid_dc] = -val.imag[valid_dc] / dc[valid_dc]
            
            reals.append(r)
            imags.append(i)
        else:
            raise ValueError(f"Harmonic {h} too high for samples {samples}")
            
    # Return formatted arrays
    if len(harmonic_list) == 1 and not has_harmonic_axis:
        return means, reals[0], imags[0]
    else:
        return means, np.stack(reals, axis=0), np.stack(imags, axis=0)


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


def evaluate_results(original_data, reconstructed_data):
    """
    Computes error metrics between original and reconstructed data.
    """
    # Create mask for valid data (exclude NaNs from thresholding)
    mask = ~np.isnan(reconstructed_data) & ~np.isnan(original_data)
    
    if mask.sum() == 0:
        return {'mse': float('nan'), 'psnr': float('nan'), 'corr': float('nan')}
        
    orig_valid = original_data[mask]
    rec_valid = reconstructed_data[mask]
    
    # MSE
    mse = np.mean((orig_valid - rec_valid)**2)
    
    # PSNR
    max_val = np.max(orig_valid)
    if mse > 0:
        psnr = 10 * np.log10(max_val**2 / mse)
    else:
        psnr = float('inf')
        
    # Correlation Coefficient
    if orig_valid.size > 1:
        corr = np.corrcoef(orig_valid.flatten(), rec_valid.flatten())[0, 1]
    else:
        corr = 0.0
        
    return {'mse': mse, 'psnr': psnr, 'corr': corr}


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # Configuration
    DATA_FILE = 'Embryo.tif'
    HARMONIC_NUM = 1
    FILTER_CONFIG = {'median_size': 3, 'median_repeat': 1, 'threshold_min': 5.0}
    
    # 1. Load Data
    print("--- Loading Data ---")
    data, meta = load_and_preprocess_data(DATA_FILE)
    print(f"Data Loaded: Shape {data.shape}, Dims {meta['dims']}")
    
    # Identify time axis (axis 0 for this dataset based on shape (56, 128, 128))
    time_axis = 0
    num_samples = data.shape[time_axis]
    
    # 2. Forward Operator (Phasor Transform)
    print("--- Running Forward Operator ---")
    mean_p, real_p, imag_p = forward_operator(data, axis=time_axis, harmonic=HARMONIC_NUM)
    print(f"Phasor Mean Shape: {mean_p.shape}")
    
    # 3. Run Inversion (Filtering + Reconstruction)
    print("--- Running Inversion (Filter + Reconstruction) ---")
    reconstructed_signal = run_inversion(
        mean_p, real_p, imag_p, 
        original_axis=time_axis, 
        original_samples=num_samples, 
        harmonic=HARMONIC_NUM,
        filter_params=FILTER_CONFIG
    )
    print(f"Reconstructed Shape: {reconstructed_signal.shape}")
    
    # 4. Evaluate
    print("--- Evaluating Results ---")
    metrics = evaluate_results(data, reconstructed_signal)
    
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"Correlation: {metrics['corr']:.4f}")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")