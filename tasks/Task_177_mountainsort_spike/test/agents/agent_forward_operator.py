import numpy as np

import matplotlib

matplotlib.use("Agg")

def forward_operator(templates, spike_times, spike_labels, n_samples, num_channels, noise_std=None):
    """
    Place templates at spike times → multi-channel recording.
    
    Forward model: multi-channel recording = Σᵢ (template_i convolved at spike_times_i) + noise
    
    Args:
        templates: (num_units, template_samples, num_channels) array
        spike_times: array of spike time indices
        spike_labels: array of unit labels for each spike
        n_samples: total number of samples in recording
        num_channels: number of channels
        noise_std: standard deviation of additive Gaussian noise (None for no noise)
    
    Returns:
        recording: (n_samples, num_channels) multi-channel recording
    """
    template_samples = templates.shape[1]
    recording = np.zeros((n_samples, num_channels))
    
    for t_idx, label in zip(spike_times, spike_labels):
        end = t_idx + template_samples
        if end <= n_samples:
            recording[t_idx:end, :] += templates[label]
    
    if noise_std is not None:
        recording += np.random.randn(*recording.shape) * noise_std
    
    return recording
