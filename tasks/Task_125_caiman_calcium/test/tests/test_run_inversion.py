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
from scipy.ndimage import gaussian_filter1d


# === Inject the Referee (evaluate_results) verbatim ===
def evaluate_results(
    true_spikes: np.ndarray,
    estimated_spikes: np.ndarray,
    fluorescence_noisy: np.ndarray,
    reconstructed_fluorescence: np.ndarray,
    spike_tol_frames: int = 2,
    results_dir: str = None
) -> dict:
    """
    Evaluate deconvolution results and create visualizations.
    """
    
    def compute_psnr(reference, estimate):
        """Compute PSNR between reference and estimate signals."""
        mse = np.mean((reference - estimate) ** 2)
        if mse < 1e-12:
            return 100.0
        data_range = np.max(reference) - np.min(reference)
        return 20 * np.log10(data_range / np.sqrt(mse))
    
    def compute_correlation(a, b):
        """Compute Pearson correlation coefficient."""
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        num = np.sum(a_centered * b_centered)
        den = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
        if den < 1e-12:
            return 0.0
        return num / den
    
    def compute_spike_detection_metrics(true_spk, est_spk, tolerance=2):
        """Compute spike detection precision, recall, and F1 score."""
        true_locs = np.where(true_spk > 0.3)[0]
        est_locs = np.where(est_spk > np.max(est_spk) * 0.15)[0]
        
        if len(true_locs) == 0 and len(est_locs) == 0:
            return 1.0, 1.0, 1.0
        if len(true_locs) == 0:
            return 0.0, 1.0, 0.0
        if len(est_locs) == 0:
            return 1.0, 0.0, 0.0
        
        true_positives = 0
        matched_est = set()
        for t_loc in true_locs:
            for e_idx, e_loc in enumerate(est_locs):
                if abs(t_loc - e_loc) <= tolerance and e_idx not in matched_est:
                    true_positives += 1
                    matched_est.add(e_idx)
                    break
        
        precision = true_positives / len(est_locs) if len(est_locs) > 0 else 0.0
        recall = true_positives / len(true_locs) if len(true_locs) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        
        return precision, recall, f1
    
    n_neurons = true_spikes.shape[0]
    
    psnr_list = []
    cc_spike_list = []
    cc_fluor_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(n_neurons):
        # PSNR on fluorescence reconstruction
        psnr = compute_psnr(fluorescence_noisy[i], reconstructed_fluorescence[i])
        psnr_list.append(psnr)
        
        # Correlation on spike trains (smoothed for better comparison)
        true_smooth = gaussian_filter1d(true_spikes[i], sigma=3)
        est_smooth = gaussian_filter1d(estimated_spikes[i], sigma=3)
        cc_spk = compute_correlation(true_smooth, est_smooth)
        cc_spike_list.append(cc_spk)
        
        # Correlation on fluorescence
        cc_fl = compute_correlation(fluorescence_noisy[i], reconstructed_fluorescence[i])
        cc_fluor_list.append(cc_fl)
        
        # Spike detection metrics
        prec, rec, f1 = compute_spike_detection_metrics(
            true_spikes[i], estimated_spikes[i], tolerance=spike_tol_frames
        )
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
        print(f"  Neuron {i+1}: PSNR={psnr:.2f} dB, CC_spike={cc_spk:.4f}, "
              f"CC_fluor={cc_fl:.4f}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    
    mean_psnr = np.mean(psnr_list)
    mean_cc_spike = np.mean(cc_spike_list)
    mean_cc_fluor = np.mean(cc_fluor_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    
    print(f"\n  === Average Metrics ===")
    print(f"  PSNR (fluorescence): {mean_psnr:.2f} dB")
    print(f"  CC (spike trains):   {mean_cc_spike:.4f}")
    print(f"  CC (fluorescence):   {mean_cc_fluor:.4f}")
    print(f"  Spike Precision:     {mean_precision:.4f}")
    print(f"  Spike Recall:        {mean_recall:.4f}")
    print(f"  Spike F1 Score:      {mean_f1:.4f}")
    
    metrics = {
        'mean_psnr': mean_psnr,
        'mean_cc_spike': mean_cc_spike,
        'mean_cc_fluor': mean_cc_fluor,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'per_neuron_psnr': psnr_list,
        'per_neuron_cc_spike': cc_spike_list,
        'per_neuron_cc_fluor': cc_fluor_list,
        'per_neuron_precision': precision_list,
        'per_neuron_recall': recall_list,
        'per_neuron_f1': f1_list,
    }
    
    return metrics


def main():
    # === Data paths ===
    data_paths = ['/data/yjh/caiman_calcium_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # === Classify files into outer and inner ===
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # === Load outer data ===
    if not outer_files:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"\nLoading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    
    # === Run agent's run_inversion ===
    print("\n--- Running agent's run_inversion ---")
    try:
        agent_output = run_inversion(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Agent's run_inversion raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # === Determine execution pattern ===
    if inner_files:
        # Pattern 2: Chained execution
        print("\n--- Pattern 2: Chained Execution ---")
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # agent_output should be callable
        if not callable(agent_output):
            print("ERROR: Agent output is not callable for chained execution.")
            sys.exit(1)
        
        try:
            final_agent_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Agent's inner call raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("\n--- Pattern 1: Direct Execution ---")
        final_agent_result = agent_output
        std_result = std_output
    
    # === Extract data for evaluation ===
    # The function returns a dict with 'estimated_spikes' and 'reconstructed_fluorescence'
    # We need true_spikes for evaluation. Check if it's in the stored data or input args.
    
    # The evaluate_results function needs: true_spikes, estimated_spikes, fluorescence_noisy, reconstructed_fluorescence
    # true_spikes is NOT returned by run_inversion - it must be part of the test data or we need to
    # look for it in the saved data structure.
    
    # fluorescence_noisy is the first positional argument to run_inversion
    fluorescence_noisy = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('fluorescence_noisy', None)
    
    if fluorescence_noisy is None:
        print("ERROR: Could not find fluorescence_noisy in inputs.")
        sys.exit(1)
    
    # Extract agent results
    agent_estimated_spikes = final_agent_result['estimated_spikes']
    agent_reconstructed_fluorescence = final_agent_result['reconstructed_fluorescence']
    
    # Extract standard results
    std_estimated_spikes = std_result['estimated_spikes']
    std_reconstructed_fluorescence = std_result['reconstructed_fluorescence']
    
    # For true_spikes: we use the standard estimated_spikes as the "ground truth" reference
    # since we don't have the actual true spike trains. This way we compare agent vs standard.
    # However, the evaluate_results function is designed to compare against true spikes.
    # We'll use std_estimated_spikes as the reference "true_spikes" for both evaluations
    # to see how well the agent matches the standard.
    
    # Actually, let's check if there's additional data in the pkl that might contain true_spikes
    print(f"\nOuter data keys: {list(outer_data.keys())}")
    
    # Look for true_spikes in outer_data
    true_spikes = None
    if 'true_spikes' in outer_data:
        true_spikes = outer_data['true_spikes']
    
    # If not found, use std_estimated_spikes as the reference
    # We evaluate BOTH agent and standard using the same true_spikes reference
    # If true_spikes is not available, we use the standard output as reference
    
    if true_spikes is None:
        # Use standard's estimated spikes as the "true" reference
        # This means standard will get perfect scores and agent will be compared against it
        print("\nNo true_spikes found in data. Using standard output as reference.")
        print("Evaluating agent output vs standard output directly...")
        
        # Strategy: evaluate both with std_estimated_spikes as ground truth
        true_spikes_ref = std_estimated_spikes
        
        print("\n=== Agent Evaluation (against standard reference) ===")
        try:
            agent_metrics = evaluate_results(
                true_spikes=true_spikes_ref,
                estimated_spikes=agent_estimated_spikes,
                fluorescence_noisy=fluorescence_noisy,
                reconstructed_fluorescence=agent_reconstructed_fluorescence
            )
        except Exception as e:
            print(f"ERROR during agent evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\n=== Standard Evaluation (against itself = perfect) ===")
        try:
            std_metrics = evaluate_results(
                true_spikes=true_spikes_ref,
                estimated_spikes=std_estimated_spikes,
                fluorescence_noisy=fluorescence_noisy,
                reconstructed_fluorescence=std_reconstructed_fluorescence
            )
        except Exception as e:
            print(f"ERROR during standard evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\nFound true_spikes with shape: {true_spikes.shape}")
        
        print("\n=== Agent Evaluation ===")
        try:
            agent_metrics = evaluate_results(
                true_spikes=true_spikes,
                estimated_spikes=agent_estimated_spikes,
                fluorescence_noisy=fluorescence_noisy,
                reconstructed_fluorescence=agent_reconstructed_fluorescence
            )
        except Exception as e:
            print(f"ERROR during agent evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\n=== Standard Evaluation ===")
        try:
            std_metrics = evaluate_results(
                true_spikes=true_spikes,
                estimated_spikes=std_estimated_spikes,
                fluorescence_noisy=fluorescence_noisy,
                reconstructed_fluorescence=std_reconstructed_fluorescence
            )
        except Exception as e:
            print(f"ERROR during standard evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # === Compare metrics ===
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    metrics_to_compare = ['mean_psnr', 'mean_cc_spike', 'mean_cc_fluor', 'mean_precision', 'mean_recall', 'mean_f1']
    
    all_pass = True
    for metric_name in metrics_to_compare:
        agent_val = agent_metrics[metric_name]
        std_val = std_metrics[metric_name]
        
        # All these metrics are "higher is better"
        # Allow 10% margin of degradation
        if std_val > 0:
            ratio = agent_val / std_val
            threshold = 0.90  # agent must be at least 90% of standard
            passed = ratio >= threshold
        elif std_val == 0 and agent_val >= 0:
            passed = True
            ratio = float('inf') if agent_val > 0 else 1.0
        else:
            # std_val < 0 (can happen with correlation)
            # agent should be >= std_val * 1.1 (more negative is worse)
            passed = agent_val >= std_val * 1.1 if std_val < 0 else agent_val >= std_val * 0.9
            ratio = agent_val / std_val if abs(std_val) > 1e-12 else float('inf')
        
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        
        print(f"  {metric_name:20s}: Agent={agent_val:+.4f}, Std={std_val:+.4f}, Ratio={ratio:.4f} [{status}]")
    
    print("=" * 60)
    
    # Primary composite score: weighted combination
    agent_score = (agent_metrics['mean_cc_spike'] + agent_metrics['mean_f1'] + agent_metrics['mean_cc_fluor']) / 3.0
    std_score = (std_metrics['mean_cc_spike'] + std_metrics['mean_f1'] + std_metrics['mean_cc_fluor']) / 3.0
    
    print(f"\nComposite Scores -> Agent: {agent_score:.4f}, Standard: {std_score:.4f}")
    
    # Final decision: agent composite score should be at least 90% of standard
    if std_score > 0:
        final_pass = agent_score >= std_score * 0.90
    else:
        final_pass = agent_score >= std_score
    
    if final_pass:
        print("\n*** RESULT: PASS - Agent performance is acceptable. ***")
        sys.exit(0)
    else:
        print("\n*** RESULT: FAIL - Agent performance degraded significantly. ***")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)