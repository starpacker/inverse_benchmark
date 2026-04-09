import matplotlib

matplotlib.use('Agg')

import numpy as np

def detect_spikes_from_deconv(s, threshold_factor=0.3):
    """
    Detect discrete spike events from deconvolved trace.
    """
    if np.max(s) <= 0:
        return np.array([], dtype=int)
    
    threshold = threshold_factor * np.max(s)
    abs_threshold = np.percentile(s[s > 0], 50) if np.any(s > 0) else 0
    threshold = max(threshold, abs_threshold)
    
    spike_frames = np.where(s > threshold)[0]
    return spike_frames
