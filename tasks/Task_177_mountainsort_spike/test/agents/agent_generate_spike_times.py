import numpy as np

import matplotlib

matplotlib.use("Agg")

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
