import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

import os

import json

def evaluate_results(pred_labels, pred_times, gt_labels, gt_times, n_neurons,
                     recording, filtered, features, params, results_dir, tolerance=15):
    """
    Evaluate spike sorting results and generate visualizations.
    
    Args:
        pred_labels: Predicted neuron labels
        pred_times: Detected spike times
        gt_labels: Ground truth neuron labels
        gt_times: Ground truth spike times
        n_neurons: Number of neurons
        recording: Raw recording
        filtered: Filtered recording
        features: PCA features
        params: Parameters dictionary
        results_dir: Directory to save results
        tolerance: Tolerance for spike matching (samples)
    
    Returns:
        metrics: Dictionary of evaluation metrics
        mapping: Cluster to neuron mapping
    """
    fs = params['fs']
    
    # ====== MATCH CLUSTERS TO GT NEURONS ======
    n_clusters = len(np.unique(pred_labels)) if len(pred_labels) > 0 else 0
    cost_matrix = np.zeros((max(n_clusters, 1), n_neurons))
    
    for i, pt in enumerate(pred_times):
        dists = np.abs(gt_times.astype(float) - float(pt))
        if len(dists) > 0 and np.min(dists) < tolerance:
            gt_idx = np.argmin(dists)
            gt_label = gt_labels[gt_idx]
            pred_label = pred_labels[i]
            if pred_label < cost_matrix.shape[0]:
                cost_matrix[pred_label, gt_label] += 1
    
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c
    
    # ====== COMPUTE METRICS ======
    correct = 0
    detected = 0
    
    for i, pt in enumerate(pred_times):
        dists = np.abs(gt_times.astype(float) - float(pt))
        if len(dists) > 0 and np.min(dists) < tolerance:
            gt_idx = np.argmin(dists)
            detected += 1
            pred_neuron = mapping.get(pred_labels[i], -1)
            if pred_neuron == gt_labels[gt_idx]:
                correct += 1
    
    accuracy = correct / max(detected, 1)
    detection_rate = detected / max(len(gt_times), 1)
    precision = detected / max(len(pred_times), 1)
    recall = detected / max(len(gt_times), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    metrics = {
        'accuracy': float(accuracy),
        'detection_rate': float(detection_rate),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_gt_spikes': int(len(gt_times)),
        'n_detected_spikes': int(len(pred_times)),
        'n_correct': int(correct)
    }
    
    # ====== SAVE METRICS ======
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {results_dir}/metrics.json")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_labels)
    np.save(os.path.join(results_dir, "reconstruction.npy"), pred_labels)
    
    # ====== VISUALIZATION ======
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Panel 1: Raw recording snippet (first 1 second, 3 channels)
    ax = axes[0, 0]
    t = np.arange(min(fs, len(recording))) / fs
    n_show = min(3, recording.shape[1])
    for ch in range(n_show):
        ax.plot(t, recording[:len(t), ch] + ch * 3, 'k', linewidth=0.3, alpha=0.7)
    ax.set_title('Raw Multi-Channel Recording (1s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    
    # Panel 2: PCA feature space with cluster colors
    ax = axes[0, 1]
    if len(features) > 0 and features.shape[1] >= 2:
        unique_labels = np.unique(pred_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for idx, label in enumerate(unique_labels):
            mask = pred_labels == label
            neuron_id = mapping.get(label, label)
            ax.scatter(features[mask, 0], features[mask, 1], 
                      c=[colors[idx]], alpha=0.5, s=10, label=f'Cluster {label}→Neuron {neuron_id}')
    ax.set_title('PCA Feature Space')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    
    # Panel 3: Raster plot — GT
    ax = axes[1, 0]
    colors_gt = plt.cm.tab10(np.linspace(0, 1, n_neurons))
    for neuron_id in range(n_neurons):
        mask = gt_labels == neuron_id
        times = gt_times[mask] / fs
        ax.scatter(times, np.full_like(times, neuron_id), c=[colors_gt[neuron_id]], s=2, marker='|')
    ax.set_title('Ground Truth Spike Raster')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron ID')
    ax.set_yticks(range(n_neurons))
    
    # Panel 4: Raster plot — Sorted
    ax = axes[1, 1]
    if len(pred_labels) > 0:
        for label in np.unique(pred_labels):
            mask = pred_labels == label
            times = pred_times[mask] / fs
            neuron_id = mapping.get(label, label)
            ax.scatter(times, np.full_like(times, neuron_id), 
                      c=[colors_gt[neuron_id % len(colors_gt)]], s=2, marker='|')
    ax.set_title('Sorted Spike Raster')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron ID')
    ax.set_yticks(range(n_neurons))
    
    fig.suptitle(f"Spike Sorting | Accuracy={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f} | "
                 f"Detected={metrics['n_detected_spikes']}/{metrics['n_gt_spikes']}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics, mapping
