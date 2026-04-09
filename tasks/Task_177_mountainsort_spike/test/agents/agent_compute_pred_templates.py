import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_pred_templates(recording, pred_times, pred_labels, mapping, template_samples, num_units, num_channels):
    """Compute mean waveform templates from predicted clusters."""
    half = template_samples // 2
    pred_templates = [None] * num_units
    inv_mapping = {v: k for k, v in mapping.items()}

    for gt_unit in range(num_units):
        pred_cluster = inv_mapping.get(gt_unit, None)
        if pred_cluster is None:
            pred_templates[gt_unit] = np.zeros((template_samples, num_channels))
            continue
        mask = pred_labels == pred_cluster
        indices = pred_times[mask]
        snippets = []
        for idx in indices:
            start = int(idx) - half
            end = start + template_samples
            if 0 <= start and end <= recording.shape[0]:
                snippets.append(recording[start:end, :])
        if len(snippets) > 0:
            pred_templates[gt_unit] = np.mean(snippets, axis=0)
        else:
            pred_templates[gt_unit] = np.zeros((template_samples, num_channels))
    return pred_templates
