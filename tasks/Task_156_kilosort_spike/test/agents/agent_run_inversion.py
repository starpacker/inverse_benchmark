import numpy as np

import matplotlib

matplotlib.use('Agg')

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

def run_inversion(filtered, params, threshold_factor=4.0, n_pca_components=5):
    """
    Run spike sorting inversion: detect spikes, extract waveforms, cluster.
    
    The inverse problem: Given observed recording y, find:
    - spike times t_i
    - spike labels (neuron assignments) z_i
    
    This is solved via:
    1. Threshold-based spike detection
    2. Waveform extraction
    3. PCA dimensionality reduction
    4. K-means clustering
    
    Args:
        filtered: Bandpass filtered recording (n_samples, n_channels)
        params: Dictionary containing fs, n_neurons, n_template_samples
        threshold_factor: Multiplier for detection threshold
        n_pca_components: Number of PCA components
    
    Returns:
        pred_times: Detected spike times
        pred_labels: Predicted neuron labels
        waveforms: Extracted waveforms
        features: PCA features
        pca: Fitted PCA object
    """
    fs = params['fs']
    n_neurons = params['n_neurons']
    n_template_samples = params.get('n_template_samples', 61)
    half = n_template_samples // 2
    
    # ====== SPIKE DETECTION ======
    # Multi-channel detection: sum of squares
    detection_signal = np.sum(filtered**2, axis=1)
    threshold = np.median(detection_signal) + threshold_factor * np.std(detection_signal)
    
    # Find crossings above threshold
    above = detection_signal > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    
    # Refine: find local maxima near each crossing
    spike_times = []
    refractory = int(0.001 * fs)  # 1ms
    last_spike = -refractory
    
    for c in crossings:
        window_start = max(0, c - half)
        window_end = min(len(detection_signal), c + half)
        peak_idx = window_start + np.argmax(detection_signal[window_start:window_end])
        if peak_idx - last_spike > refractory and peak_idx > half and peak_idx < len(filtered) - half:
            spike_times.append(peak_idx)
            last_spike = peak_idx
    
    pred_times = np.array(spike_times)
    
    # ====== WAVEFORM EXTRACTION ======
    n_channels = filtered.shape[1]
    waveforms = []
    valid_times = []
    for st in pred_times:
        start = st - half
        end = st + half + 1
        if start >= 0 and end <= len(filtered):
            wf = filtered[start:end, :].flatten()
            waveforms.append(wf)
            valid_times.append(st)
    
    waveforms = np.array(waveforms)
    valid_times = np.array(valid_times)
    
    if len(waveforms) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), None
    
    # ====== SPIKE SORTING (PCA + K-MEANS) ======
    n_components = min(n_pca_components, waveforms.shape[1], waveforms.shape[0])
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(waveforms)
    
    n_clusters = min(n_neurons, len(waveforms))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(features)
    
    return valid_times, pred_labels, waveforms, features, pca
