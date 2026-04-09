import numpy as np

import matplotlib

matplotlib.use("Agg")

def make_templates(num_units, template_samples, num_channels):
    """Create distinct waveform templates across channels."""
    t = np.linspace(0, 1, template_samples, endpoint=False)
    templates = np.zeros((num_units, template_samples, num_channels))

    # Unit 0 – biphasic, large on ch0-ch1
    wave0 = -np.sin(2 * np.pi * t) * np.exp(-3 * t)
    templates[0, :, 0] = wave0 * 1.0
    templates[0, :, 1] = wave0 * 0.7
    templates[0, :, 2] = wave0 * 0.15
    templates[0, :, 3] = wave0 * 0.10

    # Unit 1 – triphasic, large on ch2-ch3
    wave1 = (np.sin(3 * np.pi * t) * np.exp(-4 * t))
    templates[1, :, 0] = wave1 * 0.10
    templates[1, :, 1] = wave1 * 0.15
    templates[1, :, 2] = wave1 * 1.0
    templates[1, :, 3] = wave1 * 0.8

    # Unit 2 – monophasic negative, spread across channels
    wave2 = -np.exp(-((t - 0.25) ** 2) / (2 * 0.04 ** 2))
    templates[2, :, 0] = wave2 * 0.5
    templates[2, :, 1] = wave2 * 0.3
    templates[2, :, 2] = wave2 * 0.4
    templates[2, :, 3] = wave2 * 0.9

    return templates

def generate_spike_times(firing_rates, duration, sampling_rate, template_samples, random_seed):
    """Generate Poisson spike trains for each unit with a refractory period."""
    np.random.seed(random_seed)
    spike_times_list = []
    spike_labels_list = []
    refractory = int(0.002 * sampling_rate)  # 2 ms refractory

    for unit_id, rate in enumerate(firing_rates):
        isi = np.random.exponential(1.0 / rate, size=int(rate * duration * 2))
        times_sec = np.cumsum(isi)
        times_sec = times_sec[times_sec < duration - template_samples / sampling_rate]
        sample_indices = (times_sec * sampling_rate).astype(int)

        # Enforce refractory period
        if len(sample_indices) > 0:
            kept = [sample_indices[0]]
            for s in sample_indices[1:]:
                if s - kept[-1] >= refractory:
                    kept.append(s)
            sample_indices = np.array(kept)
        else:
            sample_indices = np.array([], dtype=int)

        spike_times_list.append(sample_indices)
        spike_labels_list.append(np.full(len(sample_indices), unit_id))

    spike_times = np.concatenate(spike_times_list)
    spike_labels = np.concatenate(spike_labels_list)
    order = np.argsort(spike_times)
    return spike_times[order], spike_labels[order]

def load_and_preprocess_data(num_units, template_samples, num_channels, 
                              firing_rates, duration, sampling_rate, 
                              noise_std, random_seed):
    """
    Synthesize ground-truth templates and spike times, then generate recording.
    
    Returns:
        templates: (num_units, template_samples, num_channels) array of waveform templates
        gt_times: array of ground truth spike times (sample indices)
        gt_labels: array of ground truth unit labels for each spike
        recording: (n_samples, num_channels) multi-channel recording with spikes + noise
        config: dictionary containing all configuration parameters
    """
    np.random.seed(random_seed)
    
    # Create templates
    templates = make_templates(num_units, template_samples, num_channels)
    
    # Generate ground truth spike times
    gt_times, gt_labels = generate_spike_times(
        firing_rates, duration, sampling_rate, template_samples, random_seed
    )
    
    print(f"  GT spikes: {len(gt_times)}  (units: {np.bincount(gt_labels)})")
    
    # Generate recording using forward operator
    n_samples = int(sampling_rate * duration)
    recording = np.zeros((n_samples, num_channels))
    for t_idx, label in zip(gt_times, gt_labels):
        end = t_idx + template_samples
        if end <= n_samples:
            recording[t_idx:end, :] += templates[label]
    recording += np.random.randn(*recording.shape) * noise_std
    
    # Compute SNR
    snr_per_ch = []
    for ch in range(num_channels):
        signal_power = np.var(recording[:, ch]) - noise_std ** 2
        snr_per_ch.append(10 * np.log10(max(signal_power, 1e-12) / noise_std ** 2))
    print(f"  SNR per channel (dB): {[f'{s:.1f}' for s in snr_per_ch]}")
    
    config = {
        'num_units': num_units,
        'template_samples': template_samples,
        'num_channels': num_channels,
        'firing_rates': firing_rates,
        'duration': duration,
        'sampling_rate': sampling_rate,
        'noise_std': noise_std,
        'random_seed': random_seed,
    }
    
    return templates, gt_times, gt_labels, recording, config
