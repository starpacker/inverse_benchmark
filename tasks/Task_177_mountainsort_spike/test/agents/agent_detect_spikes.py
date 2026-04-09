import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.signal import butter, sosfiltfilt, find_peaks

def detect_spikes(recording_filt, threshold_factor, sampling_rate, template_samples):
    """Detect spikes via negative threshold crossing on the channel with max energy."""
    energy = np.sum(recording_filt ** 2, axis=1)
    threshold = threshold_factor * np.median(np.abs(energy)) / 0.6745
    peaks, _ = find_peaks(energy, height=threshold,
                          distance=int(0.002 * sampling_rate))
    # Exclude edges
    margin = template_samples
    peaks = peaks[(peaks >= margin) & (peaks < recording_filt.shape[0] - margin)]
    return peaks
