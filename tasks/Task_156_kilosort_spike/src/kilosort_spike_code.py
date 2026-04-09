"""
kilosort_spike - Spike Sorting Inverse Problem
===============================================
Task: Detect and classify neural spikes from multi-channel electrophysiology
Repo: https://github.com/MouseLand/Kilosort
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import os, sys, json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_spike_template(n_samples=61, template_type=0):
    """Generate realistic biphasic spike waveform templates."""
    t = np.linspace(-1, 2, n_samples)
    if template_type == 0:
        template = -np.exp(-t**2/0.1) + 0.3*np.exp(-(t-0.5)**2/0.2)
    elif template_type == 1:
        template = -0.8*np.exp(-t**2/0.08) + 0.5*np.exp(-(t-0.4)**2/0.15)
    elif template_type == 2:
        template = -1.2*np.exp(-t**2/0.12) + 0.2*np.exp(-(t-0.6)**2/0.25)
    else:
        template = -0.6*np.exp(-t**2/0.06) + 0.4*np.exp(-(t-0.3)**2/0.1)
    return template / np.abs(template).max()

def synthesize_recording(n_neurons=4, n_channels=8, fs=30000, duration=10.0, 
                         spike_rate=3.0, noise_level=0.3, seed=42):
    """Synthesize multi-channel recording with known spike times."""
    np.random.seed(seed)
    n_samples = int(fs * duration)
    n_template_samples = 61  # ~2ms at 30kHz
    half_template = n_template_samples // 2
    
    recording = np.random.randn(n_samples, n_channels) * noise_level
    
    gt_spike_times = []
    gt_spike_labels = []
    templates = []
    spatial_profiles = []
    
    for neuron_id in range(n_neurons):
        # Template
        template = generate_spike_template(n_template_samples, neuron_id)
        amplitude = 1.0 + 0.5 * neuron_id
        template *= amplitude
        templates.append(template)
        
        # Spatial profile (different for each neuron)
        spatial = np.random.rand(n_channels)
        peak_ch = neuron_id % n_channels
        for ch in range(n_channels):
            dist = min(abs(ch - peak_ch), n_channels - abs(ch - peak_ch))
            spatial[ch] = np.exp(-dist**2 / 2.0)
        spatial /= spatial.max()
        spatial_profiles.append(spatial)
        
        # Spike times (Poisson process with refractory period)
        n_spikes_expected = int(spike_rate * duration)
        isi = np.random.exponential(fs / spike_rate, n_spikes_expected * 2)
        isi = np.maximum(isi, int(0.002 * fs))  # 2ms refractory
        spike_times = np.cumsum(isi).astype(int)
        spike_times = spike_times[spike_times < n_samples - n_template_samples]
        spike_times = spike_times[:n_spikes_expected]
        
        for st in spike_times:
            start = st - half_template
            end = st + half_template + 1
            if start >= 0 and end <= n_samples:
                for ch in range(n_channels):
                    recording[start:end, ch] += template * spatial[ch]
                gt_spike_times.append(st)
                gt_spike_labels.append(neuron_id)
    
    sort_idx = np.argsort(gt_spike_times)
    gt_spike_times = np.array(gt_spike_times)[sort_idx]
    gt_spike_labels = np.array(gt_spike_labels)[sort_idx]
    
    return recording, gt_spike_times, gt_spike_labels, templates, spatial_profiles

def bandpass_filter(data, low=300, high=6000, fs=30000, order=3):
    """Apply bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def detect_spikes(filtered, fs=30000, threshold_factor=4.0, n_template_samples=61):
    """Detect spikes by threshold crossing on peak channel."""
    half = n_template_samples // 2
    # Use channel with highest variance
    energy = np.std(filtered, axis=0)
    
    # Multi-channel detection: sum of squares
    detection_signal = np.sum(filtered**2, axis=1)
    threshold = np.median(detection_signal) + threshold_factor * np.std(detection_signal)
    
    # Find peaks above threshold
    above = detection_signal > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    
    # Refine: find local minima (negative peak) near each crossing
    spike_times = []
    refractory = int(0.001 * fs)  # 1ms
    last_spike = -refractory
    
    for c in crossings:
        window_start = max(0, c - half)
        window_end = min(len(detection_signal), c + half)
        peak_idx = window_start + np.argmax(detection_signal[window_start:window_end])
        if peak_idx - last_spike > refractory and peak_idx > half and peak_idx < len(filtered) - half:
            spike_times.append(peak_idx)
            last_spike = peak_idx
    
    return np.array(spike_times)

def extract_waveforms(filtered, spike_times, n_template_samples=61):
    """Extract waveform snippets around detected spikes."""
    half = n_template_samples // 2
    n_channels = filtered.shape[1]
    waveforms = []
    valid_times = []
    for st in spike_times:
        start = st - half
        end = st + half + 1
        if start >= 0 and end <= len(filtered):
            wf = filtered[start:end, :].flatten()
            waveforms.append(wf)
            valid_times.append(st)
    return np.array(waveforms), np.array(valid_times)

def sort_spikes(waveforms, n_clusters, n_pca_components=5):
    """Sort spikes using PCA + KMeans."""
    pca = PCA(n_components=min(n_pca_components, waveforms.shape[1], waveforms.shape[0]))
    features = pca.fit_transform(waveforms)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return labels, features, pca

def match_clusters(pred_labels, pred_times, gt_labels, gt_times, n_neurons, tolerance=15):
    """Match predicted clusters to GT neurons using Hungarian algorithm."""
    n_clusters = len(np.unique(pred_labels))
    cost_matrix = np.zeros((n_clusters, n_neurons))
    
    for i, pt in enumerate(pred_times):
        dists = np.abs(gt_times.astype(float) - float(pt))
        if len(dists) > 0 and np.min(dists) < tolerance:
            gt_idx = np.argmin(dists)
            gt_label = gt_labels[gt_idx]
            pred_label = pred_labels[i]
            cost_matrix[pred_label, gt_label] += 1
    
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c
    
    return mapping

def evaluate_sorting(pred_labels, pred_times, gt_labels, gt_times, mapping, tolerance=15):
    """Evaluate spike sorting accuracy."""
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
    
    return {
        'accuracy': float(accuracy),
        'detection_rate': float(detection_rate),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_gt_spikes': int(len(gt_times)),
        'n_detected_spikes': int(len(pred_times)),
        'n_correct': int(correct)
    }

def visualize_results(recording, filtered, gt_times, gt_labels, pred_times, pred_labels, 
                      features, mapping, metrics, fs=30000, save_path='results/reconstruction_result.png'):
    """Generate visualization."""
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
    if features.shape[1] >= 2:
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(pred_labels))))
        for label in np.unique(pred_labels):
            mask = pred_labels == label
            neuron_id = mapping.get(label, label)
            ax.scatter(features[mask, 0], features[mask, 1], 
                      c=[colors[label]], alpha=0.5, s=10, label=f'Cluster {label}→Neuron {neuron_id}')
    ax.set_title('PCA Feature Space')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    
    # Panel 3: Raster plot — GT
    ax = axes[1, 0]
    n_neurons = len(np.unique(gt_labels))
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
    for label in np.unique(pred_labels):
        mask = pred_labels == label
        times = pred_times[mask] / fs
        neuron_id = mapping.get(label, label)
        ax.scatter(times, np.full_like(times, neuron_id), c=[colors_gt[neuron_id % len(colors_gt)]], s=2, marker='|')
    ax.set_title('Sorted Spike Raster')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron ID')
    ax.set_yticks(range(n_neurons))
    
    fig.suptitle(f"Spike Sorting | Accuracy={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f} | "
                 f"Detected={metrics['n_detected_spikes']}/{metrics['n_gt_spikes']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("  kilosort_spike — Spike Sorting Pipeline")
    print("=" * 60)
    
    N_NEURONS = 4
    N_CHANNELS = 8
    FS = 30000
    DURATION = 10.0
    
    # Generate synthetic data
    print("[DATA] Synthesizing multi-channel recording...")
    recording, gt_times, gt_labels, templates, spatial_profiles = synthesize_recording(
        n_neurons=N_NEURONS, n_channels=N_CHANNELS, fs=FS, duration=DURATION)
    print(f"[DATA] Recording: {recording.shape}, GT spikes: {len(gt_times)}")
    
    # Bandpass filter
    print("[FILT] Bandpass filtering (300-6000 Hz)...")
    filtered = bandpass_filter(recording, fs=FS)
    
    # Detect spikes
    print("[DET] Detecting spikes...")
    pred_times = detect_spikes(filtered, fs=FS)
    print(f"[DET] Detected {len(pred_times)} spikes (GT: {len(gt_times)})")
    
    # Extract waveforms
    waveforms, valid_times = extract_waveforms(filtered, pred_times)
    print(f"[EXT] Extracted {len(waveforms)} waveforms")
    
    # Sort spikes
    print("[SORT] Clustering spikes...")
    pred_labels, features, pca = sort_spikes(waveforms, n_clusters=N_NEURONS)
    
    # Match clusters to GT
    mapping = match_clusters(pred_labels, valid_times, gt_labels, gt_times, N_NEURONS)
    print(f"[MATCH] Cluster mapping: {mapping}")
    
    # Evaluate
    metrics = evaluate_sorting(pred_labels, valid_times, gt_labels, gt_times, mapping)
    print(f"[EVAL] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[EVAL] F1: {metrics['f1']:.4f}")
    print(f"[EVAL] Detection rate: {metrics['detection_rate']:.4f}")
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → results/metrics.json")
    
    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_labels)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), pred_labels)
    
    # Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(recording, filtered, gt_times, gt_labels, valid_times, 
                     pred_labels, features, mapping, metrics, fs=FS, save_path=vis_path)
    
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
