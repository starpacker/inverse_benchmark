import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

def match_clusters(gt_labels, pred_labels):
    """Hungarian matching of predicted clusters to GT units."""
    gt_units = np.unique(gt_labels)
    pred_units = np.unique(pred_labels)
    cost = np.zeros((len(gt_units), len(pred_units)))
    for i, gu in enumerate(gt_units):
        for j, pu in enumerate(pred_units):
            cost[i, j] = -np.sum((gt_labels == gu) & (pred_labels == pu))
    row_ind, col_ind = linear_sum_assignment(cost)
    return {pred_units[c]: gt_units[r] for r, c in zip(row_ind, col_ind)}

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

def visualize(recording, gt_times, gt_labels, pred_times, pred_labels,
              templates, pred_templates, method, metrics, save_path,
              sampling_rate, duration, num_units, num_channels, template_samples):
    """4-panel figure: raw traces, GT raster, predicted raster, templates."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    show_samples = int(0.5 * sampling_rate)  # first 0.5 s
    time_axis = np.arange(show_samples) / sampling_rate * 1000  # ms

    # Panel 1: Raw multi-channel recording
    ax = axes[0, 0]
    for ch in range(num_channels):
        offset = ch * 1.5
        ax.plot(time_axis, recording[:show_samples, ch] + offset,
                linewidth=0.4, color="k", alpha=0.7)
        ax.text(-5, offset, f"Ch{ch}", fontsize=8, va="center")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channels (offset)")
    ax.set_title("Raw Multi-Channel Recording (first 0.5 s)")
    ax.set_xlim(0, time_axis[-1])

    # Panel 2: GT spike raster
    ax = axes[0, 1]
    for u in range(num_units):
        mask = gt_labels == u
        t_sec = gt_times[mask] / sampling_rate
        ax.scatter(t_sec, np.full(np.sum(mask), u), s=2, color=colors[u],
                   label=f"Unit {u}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit ID")
    ax.set_title("Ground Truth Spike Raster")
    ax.set_yticks(range(num_units))
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xlim(0, duration)

    # Panel 3: Predicted spike raster
    ax = axes[1, 0]
    unique_pred = np.unique(pred_labels)
    for i, u in enumerate(unique_pred):
        mask = pred_labels == u
        t_sec = pred_times[mask] / sampling_rate
        ax.scatter(t_sec, np.full(np.sum(mask), i), s=2,
                   color=colors[i % len(colors)], label=f"Cluster {u}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cluster ID")
    ax.set_title(f"Sorted Spikes ({method}) | Acc={metrics['accuracy']:.2%}")
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xlim(0, duration)

    # Panel 4: Mean templates comparison
    ax = axes[1, 1]
    t_axis = np.arange(template_samples) / sampling_rate * 1000
    for u in range(min(num_units, len(pred_templates))):
        # GT template (channel with max amplitude)
        gt_ch = np.argmax(np.max(np.abs(templates[u]), axis=0))
        ax.plot(t_axis, templates[u, :, gt_ch], color=colors[u],
                linestyle="--", linewidth=2, label=f"GT Unit {u} (ch{gt_ch})")
        # Predicted mean template (same channel)
        if u < len(pred_templates) and pred_templates[u] is not None:
            ax.plot(t_axis, pred_templates[u][:, gt_ch], color=colors[u],
                    linewidth=1.5, alpha=0.8, label=f"Pred Cluster→Unit {u}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Template Comparison (GT dashed vs Predicted solid)")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved to {save_path}")

def evaluate_results(gt_times, gt_labels, pred_times, pred_labels, recording, 
                     templates, method, config, results_dir):
    """
    Match predicted spikes to GT spikes within tolerance, compute metrics, 
    visualize results, and save outputs.
    
    Args:
        gt_times: array of ground truth spike times
        gt_labels: array of ground truth unit labels
        pred_times: array of predicted spike times
        pred_labels: array of predicted cluster labels
        recording: (n_samples, num_channels) multi-channel recording
        templates: (num_units, template_samples, num_channels) ground truth templates
        method: string indicating inversion method used
        config: dictionary with configuration parameters
        results_dir: directory to save results
    
    Returns:
        metrics: dictionary containing evaluation metrics
    """
    sampling_rate = config['sampling_rate']
    duration = config['duration']
    num_units = config['num_units']
    num_channels = config['num_channels']
    template_samples = config['template_samples']
    
    tolerance_ms = 1.0
    tol_samples = int(tolerance_ms * sampling_rate / 1000)

    # Match each predicted spike to nearest GT spike
    matched_gt_labels = []
    matched_pred_labels = []
    gt_matched_mask = np.zeros(len(gt_times), dtype=bool)

    for p_idx, p_time in enumerate(pred_times):
        dists = np.abs(gt_times.astype(int) - int(p_time))
        if len(dists) > 0:
            nearest = np.argmin(dists)
            if dists[nearest] <= tol_samples and not gt_matched_mask[nearest]:
                gt_matched_mask[nearest] = True
                matched_gt_labels.append(gt_labels[nearest])
                matched_pred_labels.append(pred_labels[p_idx])

    matched_gt_labels = np.array(matched_gt_labels)
    matched_pred_labels = np.array(matched_pred_labels)

    detection_rate = np.sum(gt_matched_mask) / len(gt_times) if len(gt_times) > 0 else 0.0

    if len(matched_gt_labels) == 0:
        metrics = {
            "accuracy": 0, 
            "detection_rate": detection_rate,
            "precision_per_unit": {}, 
            "recall_per_unit": {},
            "n_gt_spikes": int(len(gt_times)),
            "n_detected_spikes": int(len(pred_times)),
            "n_matched_spikes": 0,
            "cluster_mapping": {},
        }
    else:
        # Hungarian matching
        mapping = match_clusters(matched_gt_labels, matched_pred_labels)
        remapped = np.array([mapping.get(l, -1) for l in matched_pred_labels])

        accuracy = np.mean(remapped == matched_gt_labels)

        # Per-unit precision & recall
        prec, rec = {}, {}
        for u in np.unique(gt_labels):
            tp = np.sum((remapped == u) & (matched_gt_labels == u))
            fp = np.sum((remapped == u) & (matched_gt_labels != u))
            fn = np.sum((remapped != u) & (matched_gt_labels == u))
            prec[int(u)] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec[int(u)] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        metrics = {
            "accuracy": float(accuracy),
            "detection_rate": float(detection_rate),
            "precision_per_unit": prec,
            "recall_per_unit": rec,
            "n_gt_spikes": int(len(gt_times)),
            "n_detected_spikes": int(len(pred_times)),
            "n_matched_spikes": int(len(matched_gt_labels)),
            "cluster_mapping": {str(k): int(v) for k, v in mapping.items()},
        }

    # Print metrics
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Detection rate: {metrics['detection_rate']:.4f}")
    print(f"  Matched:        {metrics.get('n_matched_spikes', 0)}/{metrics.get('n_gt_spikes', 0)}")
    for u in sorted(metrics.get("precision_per_unit", {}).keys()):
        p = metrics["precision_per_unit"][u]
        r = metrics["recall_per_unit"][u]
        print(f"  Unit {u}: precision={p:.3f}, recall={r:.3f}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Compute predicted templates for visualization
    mapping = metrics.get("cluster_mapping", {})
    int_mapping = {int(k): int(v) for k, v in mapping.items()}
    pred_templates = compute_pred_templates(
        recording, pred_times, pred_labels, int_mapping, 
        template_samples, num_units, num_channels
    )

    # Visualization
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    visualize(
        recording, gt_times, gt_labels, pred_times, pred_labels,
        templates, pred_templates, method, metrics, vis_path,
        sampling_rate, duration, num_units, num_channels, template_samples
    )

    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"),
            {"spike_times": gt_times, "spike_labels": gt_labels,
             "templates": templates, "recording": recording})
    np.save(os.path.join(results_dir, "reconstruction.npy"),
            {"spike_times": pred_times, "spike_labels": pred_labels,
             "method": method})

    return metrics
