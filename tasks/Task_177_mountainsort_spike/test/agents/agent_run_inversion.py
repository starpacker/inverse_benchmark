import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.signal import butter, sosfiltfilt, find_peaks

from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture

def bandpass_filter(recording, low, high, sampling_rate):
    """Apply bandpass filter to recording."""
    sos = butter(3, [low, high], btype="band", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, recording, axis=0)

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

def extract_snippets(recording, spike_indices, template_samples):
    """Extract waveform snippets around detected spike times."""
    half_width = template_samples // 2
    snippets = []
    valid_indices = []
    for idx in spike_indices:
        start = idx - half_width
        end = idx + half_width + 1
        if start >= 0 and end <= recording.shape[0]:
            snippets.append(recording[start:end, :].ravel())
            valid_indices.append(idx)
    return np.array(snippets), np.array(valid_indices)

def cluster_spikes_mountainsort(recording, sampling_rate, num_channels):
    """Try mountainsort5 via spikeinterface."""
    try:
        import spikeinterface.core as si
        import mountainsort5 as ms5

        # Create a NumpyRecording
        rec = si.NumpyRecording(
            traces_list=[recording],
            sampling_frequency=sampling_rate,
        )
        rec.set_dummy_probe_from_locations(
            np.array([[0, i * 50] for i in range(num_channels)]).astype(float)
        )

        sorting = ms5.sorting_scheme2(
            recording=rec,
            sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_threshold=5.0,
                phase1_detect_channel_radius=200,
                phase1_detect_time_radius_msec=0.5,
            ),
        )
        unit_ids = sorting.get_unit_ids()
        ms_spike_times = []
        ms_spike_labels = []
        for uid in unit_ids:
            st = sorting.get_unit_spike_train(uid)
            ms_spike_times.append(st)
            ms_spike_labels.append(np.full(len(st), uid))
        if len(ms_spike_times) > 0:
            ms_spike_times = np.concatenate(ms_spike_times)
            ms_spike_labels = np.concatenate(ms_spike_labels)
            order = np.argsort(ms_spike_times)
            print(f"[mountainsort5] Found {len(unit_ids)} units, {len(ms_spike_times)} spikes")
            return ms_spike_times[order], ms_spike_labels[order], "mountainsort5"
        else:
            raise RuntimeError("mountainsort5 found 0 units")
    except Exception as e:
        print(f"[mountainsort5] failed: {e}. Falling back to GMM.")
        return None, None, None

def cluster_spikes_gmm(snippets, n_clusters, n_pca, random_seed):
    """PCA + GMM clustering fallback."""
    pca = PCA(n_components=min(n_pca, snippets.shape[1]))
    features = pca.fit_transform(snippets)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                          n_init=5, random_state=random_seed)
    labels = gmm.fit_predict(features)
    return labels

def run_inversion(recording, templates, config):
    """
    Full inverse pipeline: detect → extract → cluster.
    
    Inverse problem: recover spike times + unit IDs from the multi-channel voltage recording.
    
    Args:
        recording: (n_samples, num_channels) multi-channel recording
        templates: (num_units, template_samples, num_channels) ground truth templates
        config: dictionary with configuration parameters
    
    Returns:
        pred_times: array of predicted spike times (sample indices)
        pred_labels: array of predicted cluster labels
        method: string indicating which method was used ('mountainsort5' or 'gmm')
    """
    sampling_rate = config['sampling_rate']
    num_channels = config['num_channels']
    num_units = config['num_units']
    template_samples = config['template_samples']
    random_seed = config['random_seed']
    
    # Bandpass filter the recording
    rec_filt = bandpass_filter(recording, low=300, high=6000, sampling_rate=sampling_rate)

    # Try mountainsort5 first
    ms_times, ms_labels, method = cluster_spikes_mountainsort(rec_filt, sampling_rate, num_channels)
    if ms_times is not None:
        return ms_times, ms_labels, method

    # Fallback: manual detection + GMM
    detected = detect_spikes(rec_filt, threshold_factor=4.5, 
                             sampling_rate=sampling_rate, 
                             template_samples=template_samples)
    snippets, valid_detected = extract_snippets(rec_filt, detected, template_samples)
    
    if len(snippets) == 0:
        return np.array([]), np.array([]), "gmm"
    
    labels = cluster_spikes_gmm(snippets, n_clusters=num_units, n_pca=5, random_seed=random_seed)
    
    return valid_detected, labels, "gmm"
