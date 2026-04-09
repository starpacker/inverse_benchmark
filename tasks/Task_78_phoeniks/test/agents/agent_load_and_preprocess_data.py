import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def load_and_preprocess_data(example_dir, time_start=10e-12, time_end=90e-12, new_freq_resolution=2e9):
    """
    Load artificial THz-TDS data from the phoeniks example directory.
    
    Performs:
      1. Load reference and sample time-domain traces
      2. Load ground truth n, k, alpha values
      3. Apply windowing to time-domain traces
      4. Zero-pad for better frequency resolution
      5. Compute FFT to get frequency-domain transfer function
    
    Parameters:
        example_dir: path to example data directory
        time_start: start time for windowing (s)
        time_end: end time for windowing (s)
        new_freq_resolution: target frequency resolution after zero-padding (Hz)
    
    Returns:
        dict containing:
            - 'time': time array
            - 'td_reference': time-domain reference signal
            - 'td_sample': time-domain sample signal
            - 'frequency': frequency array after FFT
            - 'fd_reference': frequency-domain reference
            - 'fd_sample': frequency-domain sample
            - 'transfer_function': complex transfer function H(ω) = E_sample/E_reference
            - 'ground_truth': dict with 'frequency', 'n', 'k', 'alpha'
    """
    ref_file = os.path.join(example_dir, "Artifical_Reference.txt")
    sam_file = os.path.join(example_dir, "Artifical_Sample_1mm.txt")
    gt_file = os.path.join(example_dir, "Artifical_n_k_alpha.txt")

    ref = np.loadtxt(ref_file)
    sam = np.loadtxt(sam_file)
    gt_data = np.loadtxt(gt_file)

    time = ref[:, 0]
    td_reference = ref[:, 1]
    td_sample = sam[:, 1]

    # Ground truth: columns are [frequency, n, k, alpha]
    ground_truth = {
        'frequency': gt_data[:, 0],
        'n': gt_data[:, 1],
        'k': gt_data[:, 2],
        'alpha': gt_data[:, 3],
    }

    # Apply windowing to isolate main pulse
    dt = time[1] - time[0]
    window_mask = (time >= time_start) & (time <= time_end)
    
    # Create smooth window (Tukey-like)
    n_samples = len(time)
    window = np.zeros(n_samples)
    window[window_mask] = 1.0
    
    # Apply Hann window for smooth edges
    windowed_indices = np.where(window_mask)[0]
    if len(windowed_indices) > 0:
        hann = np.hanning(len(windowed_indices))
        window[windowed_indices] = hann
    
    td_reference_windowed = td_reference * window
    td_sample_windowed = td_sample * window

    # Zero-padding for better frequency resolution
    current_duration = time[-1] - time[0]
    current_freq_resolution = 1.0 / current_duration
    
    if new_freq_resolution < current_freq_resolution:
        new_duration = 1.0 / new_freq_resolution
        n_new_samples = int(np.ceil(new_duration / dt))
        
        # Pad to new length
        td_ref_padded = np.zeros(n_new_samples)
        td_sam_padded = np.zeros(n_new_samples)
        td_ref_padded[:len(td_reference_windowed)] = td_reference_windowed
        td_sam_padded[:len(td_sample_windowed)] = td_sample_windowed
        
        time_padded = np.arange(n_new_samples) * dt + time[0]
    else:
        td_ref_padded = td_reference_windowed
        td_sam_padded = td_sample_windowed
        time_padded = time

    # Compute FFT
    n_fft = len(td_ref_padded)
    fd_reference = np.fft.fft(td_ref_padded)
    fd_sample = np.fft.fft(td_sam_padded)
    frequency = np.fft.fftfreq(n_fft, dt)
    
    # Only positive frequencies
    pos_mask = frequency > 0
    frequency = frequency[pos_mask]
    fd_reference = fd_reference[pos_mask]
    fd_sample = fd_sample[pos_mask]
    
    # Compute transfer function
    # Add small epsilon to avoid division by zero
    epsilon = 1e-30
    transfer_function = fd_sample / (fd_reference + epsilon)

    return {
        'time': time_padded,
        'td_reference': td_ref_padded,
        'td_sample': td_sam_padded,
        'frequency': frequency,
        'fd_reference': fd_reference,
        'fd_sample': fd_sample,
        'transfer_function': transfer_function,
        'ground_truth': ground_truth,
    }
