import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
