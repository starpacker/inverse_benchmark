"""
Task 177: mountainsort_spike
Spike sorting: clustering and classifying neural spikes from multi-channel recordings.

Forward model: multi-channel recording = Σᵢ (template_i convolved at spike_times_i) + noise
Inverse problem: recover spike times + unit IDs from the multi-channel voltage recording.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ── Configuration ──────────────────────────────────────────────────────────
NUM_CHANNELS = 4
SAMPLING_RATE = 30000          # Hz
DURATION = 10.0                # seconds
NUM_UNITS = 3
TEMPLATE_SAMPLES = 45          # ~1.5 ms at 30 kHz
NOISE_STD = 0.08               # controls SNR (~10-15)
RANDOM_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

np.random.seed(RANDOM_SEED)


# ── 1. Synthesize ground-truth templates ───────────────────────────────────

def make_templates():
    """Create 3 distinct waveform templates across 4 channels."""
    t = np.linspace(0, 1, TEMPLATE_SAMPLES, endpoint=False)
    templates = np.zeros((NUM_UNITS, TEMPLATE_SAMPLES, NUM_CHANNELS))

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


def generate_spike_times(firing_rates=(8, 12, 6)):
    """Generate Poisson spike trains for each unit with a refractory period."""
    n_samples = int(SAMPLING_RATE * DURATION)
    spike_times_list = []
    spike_labels_list = []
    refractory = int(0.002 * SAMPLING_RATE)  # 2 ms refractory

    for unit_id, rate in enumerate(firing_rates):
        isi = np.random.exponential(1.0 / rate, size=int(rate * DURATION * 2))
        times_sec = np.cumsum(isi)
        times_sec = times_sec[times_sec < DURATION - TEMPLATE_SAMPLES / SAMPLING_RATE]
        sample_indices = (times_sec * SAMPLING_RATE).astype(int)

        # Enforce refractory period
        kept = [sample_indices[0]]
        for s in sample_indices[1:]:
            if s - kept[-1] >= refractory:
                kept.append(s)
        sample_indices = np.array(kept)

        spike_times_list.append(sample_indices)
        spike_labels_list.append(np.full(len(sample_indices), unit_id))

    spike_times = np.concatenate(spike_times_list)
    spike_labels = np.concatenate(spike_labels_list)
    order = np.argsort(spike_times)
    return spike_times[order], spike_labels[order]


# ── 2. Forward operator ───────────────────────────────────────────────────

def forward_operator(templates, spike_times, spike_labels):
    """Place templates at spike times → multi-channel recording."""
    n_samples = int(SAMPLING_RATE * DURATION)
    recording = np.zeros((n_samples, NUM_CHANNELS))
    for t_idx, label in zip(spike_times, spike_labels):
        end = t_idx + TEMPLATE_SAMPLES
        if end <= n_samples:
            recording[t_idx:end, :] += templates[label]
    recording += np.random.randn(*recording.shape) * NOISE_STD
    return recording


# ── 3. Inverse solver ─────────────────────────────────────────────────────

def bandpass_filter(recording, low=300, high=6000):
    sos = butter(3, [low, high], btype="band", fs=SAMPLING_RATE, output="sos")
    return sosfiltfilt(sos, recording, axis=0)


def detect_spikes(recording_filt, threshold_factor=4.5):
    """Detect spikes via negative threshold crossing on the channel with max energy."""
    energy = np.sum(recording_filt ** 2, axis=1)
    threshold = threshold_factor * np.median(np.abs(energy)) / 0.6745
    peaks, _ = find_peaks(energy, height=threshold,
                          distance=int(0.002 * SAMPLING_RATE))
    # Exclude edges
    margin = TEMPLATE_SAMPLES
    peaks = peaks[(peaks >= margin) & (peaks < recording_filt.shape[0] - margin)]
    return peaks


def extract_snippets(recording, spike_indices, half_width=None):
    if half_width is None:
        half_width = TEMPLATE_SAMPLES // 2
    snippets = []
    valid_indices = []
    for idx in spike_indices:
        start = idx - half_width
        end = idx + half_width + 1
        if start >= 0 and end <= recording.shape[0]:
            snippets.append(recording[start:end, :].ravel())
            valid_indices.append(idx)
    return np.array(snippets), np.array(valid_indices)


def cluster_spikes_mountainsort(recording, detected_indices):
    """Try mountainsort5 via spikeinterface, fallback to GMM."""
    try:
        import spikeinterface.core as si
        import mountainsort5 as ms5

        # Create a NumpyRecording
        rec = si.NumpyRecording(
            traces_list=[recording],
            sampling_frequency=SAMPLING_RATE,
        )
        rec.set_dummy_probe_from_locations(
            np.array([[0, i * 50] for i in range(NUM_CHANNELS)]).astype(float)
        )

        sorting = ms5.sorting_scheme2(
            recording=rec,
            sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_threshold=5.0,
                phase1_detect_channel_radius=200,
                phase1_detect_time_radius_msec=0.5,
            ),
        )
        unit_ids = sorting.get_unit_ids()
        ms_spike_times = []
        ms_spike_labels = []
        for uid in unit_ids:
            st = sorting.get_unit_spike_train(uid)
            ms_spike_times.append(st)
            ms_spike_labels.append(np.full(len(st), uid))
        if len(ms_spike_times) > 0:
            ms_spike_times = np.concatenate(ms_spike_times)
            ms_spike_labels = np.concatenate(ms_spike_labels)
            order = np.argsort(ms_spike_times)
            print(f"[mountainsort5] Found {len(unit_ids)} units, {len(ms_spike_times)} spikes")
            return ms_spike_times[order], ms_spike_labels[order], "mountainsort5"
        else:
            raise RuntimeError("mountainsort5 found 0 units")
    except Exception as e:
        print(f"[mountainsort5] failed: {e}. Falling back to GMM.")
        return None, None, None


def cluster_spikes_gmm(snippets, n_clusters=NUM_UNITS, n_pca=5):
    """PCA + GMM clustering fallback."""
    pca = PCA(n_components=min(n_pca, snippets.shape[1]))
    features = pca.fit_transform(snippets)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                          n_init=5, random_state=RANDOM_SEED)
    labels = gmm.fit_predict(features)
    return labels


def inverse_solve(recording, templates):
    """Full inverse pipeline: detect → extract → cluster."""
    rec_filt = bandpass_filter(recording)

    # Try mountainsort5 first
    ms_times, ms_labels, method = cluster_spikes_mountainsort(rec_filt, None)
    if ms_times is not None:
        return ms_times, ms_labels, method

    # Fallback: manual detection + GMM
    detected = detect_spikes(rec_filt)
    snippets, valid_detected = extract_snippets(rec_filt, detected)
    if len(snippets) == 0:
        return np.array([]), np.array([]), "gmm"
    labels = cluster_spikes_gmm(snippets)
    return valid_detected, labels, "gmm"


# ── 4. Evaluation ─────────────────────────────────────────────────────────

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


def evaluate(gt_times, gt_labels, pred_times, pred_labels, tolerance_ms=1.0):
    """Match predicted spikes to GT spikes within tolerance, then compute metrics."""
    tol_samples = int(tolerance_ms * SAMPLING_RATE / 1000)

    # Match each predicted spike to nearest GT spike
    matched_gt_labels = []
    matched_pred_labels = []
    gt_matched_mask = np.zeros(len(gt_times), dtype=bool)

    for p_idx, p_time in enumerate(pred_times):
        dists = np.abs(gt_times.astype(int) - int(p_time))
        nearest = np.argmin(dists)
        if dists[nearest] <= tol_samples and not gt_matched_mask[nearest]:
            gt_matched_mask[nearest] = True
            matched_gt_labels.append(gt_labels[nearest])
            matched_pred_labels.append(pred_labels[p_idx])

    matched_gt_labels = np.array(matched_gt_labels)
    matched_pred_labels = np.array(matched_pred_labels)

    detection_rate = np.sum(gt_matched_mask) / len(gt_times) if len(gt_times) > 0 else 0.0

    if len(matched_gt_labels) == 0:
        return {"accuracy": 0, "detection_rate": detection_rate,
                "precision_per_unit": {}, "recall_per_unit": {}}

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

    return {
        "accuracy": float(accuracy),
        "detection_rate": float(detection_rate),
        "precision_per_unit": prec,
        "recall_per_unit": rec,
        "n_gt_spikes": int(len(gt_times)),
        "n_detected_spikes": int(len(pred_times)),
        "n_matched_spikes": int(len(matched_gt_labels)),
        "cluster_mapping": {str(k): int(v) for k, v in mapping.items()},
    }


# ── 5. Visualization ──────────────────────────────────────────────────────

def visualize(recording, gt_times, gt_labels, pred_times, pred_labels,
              templates, pred_templates, method, metrics, save_path):
    """4-panel figure: raw traces, GT raster, predicted raster, templates."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    show_samples = int(0.5 * SAMPLING_RATE)  # first 0.5 s
    time_axis = np.arange(show_samples) / SAMPLING_RATE * 1000  # ms

    # Panel 1: Raw multi-channel recording
    ax = axes[0, 0]
    for ch in range(NUM_CHANNELS):
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
    for u in range(NUM_UNITS):
        mask = gt_labels == u
        t_sec = gt_times[mask] / SAMPLING_RATE
        ax.scatter(t_sec, np.full(np.sum(mask), u), s=2, color=colors[u],
                   label=f"Unit {u}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit ID")
    ax.set_title("Ground Truth Spike Raster")
    ax.set_yticks(range(NUM_UNITS))
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xlim(0, DURATION)

    # Panel 3: Predicted spike raster
    ax = axes[1, 0]
    unique_pred = np.unique(pred_labels)
    for i, u in enumerate(unique_pred):
        mask = pred_labels == u
        t_sec = pred_times[mask] / SAMPLING_RATE
        ax.scatter(t_sec, np.full(np.sum(mask), i), s=2,
                   color=colors[i % len(colors)], label=f"Cluster {u}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cluster ID")
    ax.set_title(f"Sorted Spikes ({method}) | Acc={metrics['accuracy']:.2%}")
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xlim(0, DURATION)

    # Panel 4: Mean templates comparison
    ax = axes[1, 1]
    t_axis = np.arange(TEMPLATE_SAMPLES) / SAMPLING_RATE * 1000
    for u in range(min(NUM_UNITS, len(pred_templates))):
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


def compute_pred_templates(recording, pred_times, pred_labels, mapping):
    """Compute mean waveform templates from predicted clusters."""
    half = TEMPLATE_SAMPLES // 2
    pred_templates = [None] * NUM_UNITS
    inv_mapping = {v: k for k, v in mapping.items()}

    for gt_unit in range(NUM_UNITS):
        pred_cluster = inv_mapping.get(gt_unit, None)
        if pred_cluster is None:
            pred_templates[gt_unit] = np.zeros((TEMPLATE_SAMPLES, NUM_CHANNELS))
            continue
        mask = pred_labels == pred_cluster
        indices = pred_times[mask]
        snippets = []
        for idx in indices:
            start = int(idx) - half
            end = start + TEMPLATE_SAMPLES
            if 0 <= start and end <= recording.shape[0]:
                snippets.append(recording[start:end, :])
        if len(snippets) > 0:
            pred_templates[gt_unit] = np.mean(snippets, axis=0)
        else:
            pred_templates[gt_unit] = np.zeros((TEMPLATE_SAMPLES, NUM_CHANNELS))
    return pred_templates


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Synthesize
    print("=== Step 1: Synthesizing data ===")
    templates = make_templates()
    gt_times, gt_labels = generate_spike_times(firing_rates=(8, 12, 6))
    print(f"  GT spikes: {len(gt_times)}  (units: {np.bincount(gt_labels)})")

    # 2. Forward operator
    print("=== Step 2: Forward operator ===")
    recording = forward_operator(templates, gt_times, gt_labels)
    snr_per_ch = []
    for ch in range(NUM_CHANNELS):
        signal_power = np.var(recording[:, ch]) - NOISE_STD ** 2
        snr_per_ch.append(10 * np.log10(max(signal_power, 1e-12) / NOISE_STD ** 2))
    print(f"  SNR per channel (dB): {[f'{s:.1f}' for s in snr_per_ch]}")

    # 3. Inverse solve
    print("=== Step 3: Inverse solve ===")
    pred_times, pred_labels, method = inverse_solve(recording, templates)
    print(f"  Method: {method}, detected {len(pred_times)} spikes")

    # 4. Evaluate
    print("=== Step 4: Evaluation ===")
    metrics = evaluate(gt_times, gt_labels, pred_times, pred_labels)
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Detection rate: {metrics['detection_rate']:.4f}")
    print(f"  Matched:        {metrics.get('n_matched_spikes', 0)}/{metrics.get('n_gt_spikes', 0)}")
    for u in sorted(metrics.get("precision_per_unit", {}).keys()):
        p = metrics["precision_per_unit"][u]
        r = metrics["recall_per_unit"][u]
        print(f"  Unit {u}: precision={p:.3f}, recall={r:.3f}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # 5. Compute predicted templates for visualization
    mapping = metrics.get("cluster_mapping", {})
    int_mapping = {int(k): int(v) for k, v in mapping.items()}
    pred_templates = compute_pred_templates(recording, pred_times, pred_labels, int_mapping)

    # 6. Visualization
    print("=== Step 5: Visualization ===")
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize(recording, gt_times, gt_labels, pred_times, pred_labels,
              templates, pred_templates, method, metrics, vis_path)

    # 7. Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"),
            {"spike_times": gt_times, "spike_labels": gt_labels,
             "templates": templates, "recording": recording})
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"),
            {"spike_times": pred_times, "spike_labels": pred_labels,
             "method": method})
    print("=== Done ===")


if __name__ == "__main__":
    main()
