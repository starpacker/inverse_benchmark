import numpy as np

import matplotlib

matplotlib.use("Agg")

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
