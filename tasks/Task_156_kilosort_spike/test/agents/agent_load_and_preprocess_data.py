import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.signal import butter, filtfilt

def generate_spike_template(n_samples=61, template_type=0):
    """Generate realistic biphasic spike waveform templates."""
    t = np.linspace(-1, 2, n_samples)
    if template_type == 0:
        template = -np.exp(-t**2/0.1) + 0.3*np.exp(-(t-0.5)**2/0.2)
    elif template_type == 1:
        template = -0.8*np.exp(-t**2/0.08) + 0.5*np.exp(-(t-0.4)**2/0.15)
    elif template_type == 2:
        template = -1.2*np.exp(-t**2/0.12) + 0.2*np.exp(-(t-0.6)**2/0.25)
    else:
        template = -0.6*np.exp(-t**2/0.06) + 0.4*np.exp(-(t-0.3)**2/0.1)
    return template / np.abs(template).max()

def load_and_preprocess_data(n_neurons=4, n_channels=8, fs=30000, duration=10.0,
                              spike_rate=3.0, noise_level=0.3, seed=42,
                              filter_low=300, filter_high=6000, filter_order=3):
    """
    Load/synthesize and preprocess multi-channel electrophysiology data.
    
    Returns:
        recording: Raw multi-channel recording (n_samples, n_channels)
        filtered: Bandpass filtered recording (n_samples, n_channels)
        gt_spike_times: Ground truth spike times
        gt_spike_labels: Ground truth spike labels (neuron IDs)
        templates: List of spike templates for each neuron
        spatial_profiles: List of spatial profiles for each neuron
        params: Dictionary of parameters
    """
    np.random.seed(seed)
    n_samples = int(fs * duration)
    n_template_samples = 61  # ~2ms at 30kHz
    half_template = n_template_samples // 2
    
    # Initialize recording with noise
    recording = np.random.randn(n_samples, n_channels) * noise_level
    
    gt_spike_times = []
    gt_spike_labels = []
    templates = []
    spatial_profiles = []
    
    for neuron_id in range(n_neurons):
        # Generate template
        template = generate_spike_template(n_template_samples, neuron_id)
        amplitude = 1.0 + 0.5 * neuron_id
        template *= amplitude
        templates.append(template)
        
        # Spatial profile (different for each neuron)
        spatial = np.random.rand(n_channels)
        peak_ch = neuron_id % n_channels
        for ch in range(n_channels):
            dist = min(abs(ch - peak_ch), n_channels - abs(ch - peak_ch))
            spatial[ch] = np.exp(-dist**2 / 2.0)
        spatial /= spatial.max()
        spatial_profiles.append(spatial)
        
        # Spike times (Poisson process with refractory period)
        n_spikes_expected = int(spike_rate * duration)
        isi = np.random.exponential(fs / spike_rate, n_spikes_expected * 2)
        isi = np.maximum(isi, int(0.002 * fs))  # 2ms refractory
        spike_times = np.cumsum(isi).astype(int)
        spike_times = spike_times[spike_times < n_samples - n_template_samples]
        spike_times = spike_times[:n_spikes_expected]
        
        for st in spike_times:
            start = st - half_template
            end = st + half_template + 1
            if start >= 0 and end <= n_samples:
                for ch in range(n_channels):
                    recording[start:end, ch] += template * spatial[ch]
                gt_spike_times.append(st)
                gt_spike_labels.append(neuron_id)
    
    # Sort by time
    sort_idx = np.argsort(gt_spike_times)
    gt_spike_times = np.array(gt_spike_times)[sort_idx]
    gt_spike_labels = np.array(gt_spike_labels)[sort_idx]
    
    # Bandpass filter
    nyq = fs / 2
    b, a = butter(filter_order, [filter_low/nyq, filter_high/nyq], btype='band')
    filtered = filtfilt(b, a, recording, axis=0)
    
    params = {
        'n_neurons': n_neurons,
        'n_channels': n_channels,
        'fs': fs,
        'duration': duration,
        'n_template_samples': n_template_samples,
        'filter_low': filter_low,
        'filter_high': filter_high
    }
    
    return recording, filtered, gt_spike_times, gt_spike_labels, templates, spatial_profiles, params
