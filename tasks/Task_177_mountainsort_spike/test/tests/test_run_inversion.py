import sys
import os
import dill
import numpy as np
import traceback
import json
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# REFEREE FUNCTION (Injected from Reference B)
# ============================================================

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
        ax.scatter(t_sec, np.full(np.sum(mask), u), s=2, color=colors[u % len(colors)],
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
        ax.plot(t_axis, templates[u, :, gt_ch], color=colors[u % len(colors)],
                linestyle="--", linewidth=2, label=f"GT Unit {u} (ch{gt_ch})")
        # Predicted mean template (same channel)
        if u < len(pred_templates) and pred_templates[u] is not None:
            ax.plot(t_axis, pred_templates[u][:, gt_ch], color=colors[u % len(colors)],
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

# ============================================================
# MAIN TEST LOGIC
# ============================================================

def main():
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    results_dir = './test_results'
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer (primary) data
        print("\n[Step 1] Loading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        func_name = outer_data['func_name']
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        std_output = outer_data['output']
        
        print(f"  Function: {func_name}")
        print(f"  Number of args: {len(args)}")
        print(f"  Kwargs keys: {list(kwargs.keys())}")
        
        # Extract inputs for later use
        recording = args[0] if len(args) > 0 else kwargs.get('recording')
        templates = args[1] if len(args) > 1 else kwargs.get('templates')
        config = args[2] if len(args) > 2 else kwargs.get('config')
        
        print(f"  Recording shape: {recording.shape}")
        print(f"  Templates shape: {templates.shape}")
        print(f"  Config: {config}")
        
        # Run the agent's implementation
        print("\n[Step 2] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution)
        if len(inner_data_paths) > 0:
            print("\n[Step 3] Chained execution detected, running inner function...")
            # Load inner data
            with open(inner_data_paths[0], 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data['args']
            inner_kwargs = inner_data['kwargs']
            std_result = inner_data['output']
            
            # Execute the returned operator
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Unpack results (pred_times, pred_labels, method)
        agent_pred_times, agent_pred_labels, agent_method = final_result
        std_pred_times, std_pred_labels, std_method = std_result
        
        print(f"\n[Step 3] Results:")
        print(f"  Agent method: {agent_method}")
        print(f"  Agent pred_times shape: {agent_pred_times.shape if hasattr(agent_pred_times, 'shape') else len(agent_pred_times)}")
        print(f"  Agent pred_labels shape: {agent_pred_labels.shape if hasattr(agent_pred_labels, 'shape') else len(agent_pred_labels)}")
        print(f"  Std method: {std_method}")
        print(f"  Std pred_times shape: {std_pred_times.shape if hasattr(std_pred_times, 'shape') else len(std_pred_times)}")
        print(f"  Std pred_labels shape: {std_pred_labels.shape if hasattr(std_pred_labels, 'shape') else len(std_pred_labels)}")
        
        # We need ground truth spike times and labels for evaluation
        # These should be generated from the same process that created the recording
        # For now, we'll use the standard output as a proxy for ground truth comparison
        
        # Generate synthetic ground truth from config
        # Based on gen_data_code, we need to recreate gt_times and gt_labels
        np.random.seed(config['random_seed'])
        sampling_rate = config['sampling_rate']
        duration = config['duration']
        num_units = config['num_units']
        firing_rate = config.get('firing_rate', 10)
        
        # Generate ground truth spike times (following the original data generation logic)
        gt_times_list = []
        gt_labels_list = []
        for unit_id in range(num_units):
            # Poisson process for spike times
            n_spikes = int(firing_rate * duration)
            spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
            spike_samples = (spike_times * sampling_rate).astype(int)
            
            # Filter out edge spikes
            template_samples = config['template_samples']
            valid_mask = (spike_samples >= template_samples) & (spike_samples < recording.shape[0] - template_samples)
            spike_samples = spike_samples[valid_mask]
            
            gt_times_list.append(spike_samples)
            gt_labels_list.append(np.full(len(spike_samples), unit_id))
        
        gt_times = np.concatenate(gt_times_list)
        gt_labels = np.concatenate(gt_labels_list)
        
        # Sort by time
        sort_idx = np.argsort(gt_times)
        gt_times = gt_times[sort_idx]
        gt_labels = gt_labels[sort_idx]
        
        print(f"\n[Step 4] Ground truth generated:")
        print(f"  GT times shape: {gt_times.shape}")
        print(f"  GT labels shape: {gt_labels.shape}")
        
        # Evaluate agent's results
        print("\n[Step 5] Evaluating agent's results...")
        agent_results_dir = os.path.join(results_dir, 'agent')
        agent_metrics = evaluate_results(
            gt_times, gt_labels, 
            agent_pred_times, agent_pred_labels,
            recording, templates, agent_method, config, 
            agent_results_dir
        )
        
        # Evaluate standard results
        print("\n[Step 6] Evaluating standard results...")
        std_results_dir = os.path.join(results_dir, 'standard')
        std_metrics = evaluate_results(
            gt_times, gt_labels,
            std_pred_times, std_pred_labels,
            recording, templates, std_method, config,
            std_results_dir
        )
        
        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        agent_accuracy = agent_metrics['accuracy']
        std_accuracy = std_metrics['accuracy']
        agent_detection = agent_metrics['detection_rate']
        std_detection = std_metrics['detection_rate']
        
        print(f"Scores -> Agent Accuracy: {agent_accuracy:.4f}, Standard Accuracy: {std_accuracy:.4f}")
        print(f"Scores -> Agent Detection: {agent_detection:.4f}, Standard Detection: {std_detection:.4f}")
        
        # Determine success (higher is better for accuracy)
        # Allow 10% margin of error
        margin = 0.10
        
        accuracy_ok = agent_accuracy >= std_accuracy * (1 - margin)
        detection_ok = agent_detection >= std_detection * (1 - margin)
        
        print(f"\nAccuracy check: {'PASS' if accuracy_ok else 'FAIL'} (agent: {agent_accuracy:.4f}, threshold: {std_accuracy * (1-margin):.4f})")
        print(f"Detection check: {'PASS' if detection_ok else 'FAIL'} (agent: {agent_detection:.4f}, threshold: {std_detection * (1-margin):.4f})")
        
        # Overall pass if both metrics are acceptable
        if accuracy_ok and detection_ok:
            print("\n✓ TEST PASSED: Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()