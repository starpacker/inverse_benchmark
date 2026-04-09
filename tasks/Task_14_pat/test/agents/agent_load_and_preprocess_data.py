import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import patato as pat

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # btype='band' creates a bandpass filter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to data along the last axis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # axis=-1 ensures we filter along the time dimension
    y = filtfilt(b, a, data, axis=-1)
    return y

def load_and_preprocess_data(filename, lp_filter=7e6, hp_filter=5e3):
    """
    Load data from HDF5 file and preprocess it.
    
    Steps:
    1. Load raw data, geometry, wavelengths, speed of sound from HDF5
    2. Apply bandpass filtering
    3. Apply Hilbert transform and extract imaginary part
    
    Returns:
        dict containing processed_data, fs, geometry, wavelengths, speed_of_sound
    """
    print(f"Loading data from {filename}...")
    # Load only the first frame to save memory/time for this tutorial
    padata = pat.PAData.from_hdf5(filename)[0:1]
    
    # Extract Time Series Data
    time_series = padata.get_time_series()
    raw_data = time_series.raw_data
    fs = time_series.attributes["fs"]
    
    # Extract Metadata
    geometry = padata.get_scan_geometry()
    wavelengths = padata.get_wavelengths()
    speed_of_sound = padata.get_speed_of_sound()
    
    print(f"Data Loaded: Shape={raw_data.shape}, FS={fs}, SOS={speed_of_sound}")
    
    # Squeeze extra dimension if present (e.g. shape [1, 1, 256, 2048] -> [1, 256, 2048])
    if raw_data.ndim == 4 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]
    
    print("Preprocessing: Bandpass Filtering...")
    # Note: hp_filter is the low frequency bound, lp_filter is the high frequency bound
    filtered_data = butter_bandpass_filter(raw_data, hp_filter, lp_filter, fs, order=4)
    
    print("Preprocessing: Envelope Detection (Hilbert)...")
    analytic_signal = hilbert(filtered_data, axis=-1)
    
    # Extract imaginary part (phase shifted signal)
    preprocessed_signal = np.imag(analytic_signal)
    
    return {
        'processed_data': preprocessed_signal,
        'fs': fs,
        'geometry': np.array(geometry),
        'wavelengths': np.array(wavelengths),
        'speed_of_sound': speed_of_sound
    }