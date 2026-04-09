import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(templates, spatial_profiles, spike_times, spike_labels, 
                     n_samples, n_channels, n_template_samples=61):
    """
    Forward model: Generate predicted recording from spike times and neuron assignments.
    
    This implements the generative model:
    y_pred[t, ch] = sum over spikes { template[neuron] * spatial[neuron, ch] }
    
    Args:
        templates: List of spike templates for each neuron
        spatial_profiles: List of spatial profiles for each neuron
        spike_times: Array of spike times
        spike_labels: Array of neuron labels for each spike
        n_samples: Number of samples in recording
        n_channels: Number of channels
        n_template_samples: Number of samples in template
    
    Returns:
        y_pred: Predicted recording (n_samples, n_channels)
    """
    half_template = n_template_samples // 2
    y_pred = np.zeros((n_samples, n_channels))
    
    for i, st in enumerate(spike_times):
        start = int(st) - half_template
        end = int(st) + half_template + 1
        if start >= 0 and end <= n_samples:
            neuron_id = int(spike_labels[i])
            if neuron_id < len(templates):
                template = templates[neuron_id]
                spatial = spatial_profiles[neuron_id]
                for ch in range(n_channels):
                    y_pred[start:end, ch] += template * spatial[ch]
    
    return y_pred
