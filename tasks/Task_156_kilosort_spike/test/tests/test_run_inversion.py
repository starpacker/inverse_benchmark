import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import json

# ============== INJECT REFEREE FUNCTION ==============
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
    os.makedirs(results_dir, exist_ok=True)
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


def main():
    """Main test function for run_inversion"""
    
    # Data paths provided
    data_paths = ['/data/yjh/kilosort_spike_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze data paths to detect execution pattern
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer (primary) data
        if outer_data_path is None:
            print("[ERROR] No primary data file found!")
            sys.exit(1)
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Loaded outer data with keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        # Run the agent's implementation
        print("[INFO] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("[INFO] Agent execution completed.")
        
        # Check if this is a chained execution
        if len(inner_data_paths) > 0:
            # Chained execution: agent_output should be callable
            print("[INFO] Detected chained execution pattern.")
            inner_path = inner_data_paths[0]
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_agent_result = agent_output
            std_result = std_output
        
        # Unpack results: (pred_times, pred_labels, waveforms, features, pca)
        # Agent results
        agent_pred_times, agent_pred_labels, agent_waveforms, agent_features, agent_pca = final_agent_result
        
        # Standard results
        std_pred_times, std_pred_labels, std_waveforms, std_features, std_pca = std_result
        
        print(f"[INFO] Agent detected {len(agent_pred_times)} spikes")
        print(f"[INFO] Standard detected {len(std_pred_times)} spikes")
        
        # For evaluation, we need ground truth data
        # The params should contain necessary info, and we need gt_labels, gt_times
        # These should be part of the test context
        
        # Extract params from kwargs or args
        if len(args) >= 2:
            filtered = args[0]
            params = args[1]
        else:
            filtered = kwargs.get('filtered', args[0] if len(args) > 0 else None)
            params = kwargs.get('params', args[1] if len(args) > 1 else {})
        
        # Get parameters
        n_neurons = params.get('n_neurons', 3)
        fs = params.get('fs', 30000)
        
        # For this test, we'll create a synthetic evaluation
        # Since we don't have ground truth, we compare agent vs standard outputs directly
        
        # Create results directory
        results_dir = './test_results_run_inversion'
        os.makedirs(results_dir, exist_ok=True)
        
        # Compare spike detection counts
        agent_spike_count = len(agent_pred_times)
        std_spike_count = len(std_pred_times)
        
        print(f"[COMPARISON] Agent spike count: {agent_spike_count}")
        print(f"[COMPARISON] Standard spike count: {std_spike_count}")
        
        # Compare labels distribution
        if len(agent_pred_labels) > 0:
            agent_unique_labels = len(np.unique(agent_pred_labels))
        else:
            agent_unique_labels = 0
            
        if len(std_pred_labels) > 0:
            std_unique_labels = len(np.unique(std_pred_labels))
        else:
            std_unique_labels = 0
        
        print(f"[COMPARISON] Agent unique clusters: {agent_unique_labels}")
        print(f"[COMPARISON] Standard unique clusters: {std_unique_labels}")
        
        # Compare feature shapes
        agent_feature_shape = agent_features.shape if len(agent_features) > 0 else (0, 0)
        std_feature_shape = std_features.shape if len(std_features) > 0 else (0, 0)
        
        print(f"[COMPARISON] Agent features shape: {agent_feature_shape}")
        print(f"[COMPARISON] Standard features shape: {std_feature_shape}")
        
        # Compute similarity metrics
        # 1. Spike count ratio
        if std_spike_count > 0:
            spike_ratio = agent_spike_count / std_spike_count
        else:
            spike_ratio = 1.0 if agent_spike_count == 0 else 0.0
        
        print(f"[METRIC] Spike count ratio (agent/std): {spike_ratio:.4f}")
        
        # 2. Compare spike times overlap
        if len(agent_pred_times) > 0 and len(std_pred_times) > 0:
            tolerance = 15  # samples
            matched = 0
            for at in agent_pred_times:
                dists = np.abs(std_pred_times.astype(float) - float(at))
                if np.min(dists) < tolerance:
                    matched += 1
            overlap_ratio = matched / max(len(agent_pred_times), len(std_pred_times))
        else:
            overlap_ratio = 1.0 if (len(agent_pred_times) == 0 and len(std_pred_times) == 0) else 0.0
        
        print(f"[METRIC] Spike time overlap ratio: {overlap_ratio:.4f}")
        
        # 3. Cluster count match
        cluster_match = 1.0 if agent_unique_labels == std_unique_labels else 0.5
        
        print(f"[METRIC] Cluster count match: {cluster_match:.4f}")
        
        # Compute overall score (higher is better)
        # Weight: 40% spike count, 40% overlap, 20% cluster match
        overall_score = 0.4 * min(spike_ratio, 1.0/max(spike_ratio, 0.01)) + \
                       0.4 * overlap_ratio + \
                       0.2 * cluster_match
        
        print(f"\n[RESULT] Overall performance score: {overall_score:.4f}")
        
        # Determine success (threshold at 0.7 for acceptable performance)
        # Also check if spike detection is reasonable (within 50% margin)
        threshold = 0.7
        spike_margin = 0.5
        
        spike_acceptable = (spike_ratio >= spike_margin and spike_ratio <= 1.0/spike_margin)
        score_acceptable = overall_score >= threshold
        
        if spike_acceptable and score_acceptable:
            print(f"[SUCCESS] Performance is acceptable (score={overall_score:.4f} >= {threshold})")
            print(f"[SUCCESS] Spike detection ratio is within margin ({spike_ratio:.4f})")
            sys.exit(0)
        else:
            if not spike_acceptable:
                print(f"[FAIL] Spike detection ratio out of margin ({spike_ratio:.4f})")
            if not score_acceptable:
                print(f"[FAIL] Performance score below threshold ({overall_score:.4f} < {threshold})")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()