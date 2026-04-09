import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal

# Import the target function
from agent_run_inversion import run_inversion

# Set random seed for reproducibility
np.random.seed(42)

# Setup results directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the evaluate_results function (Reference B)
def evaluate_results(data, result):
    """
    Evaluate identified modal parameters against ground truth.
    Compute frequency errors, damping errors, and MAC values.
    Generate visualization and save metrics.
    """
    n_dof = data["n_dof"]
    freq_true = data["freq_true"]
    damping_ratios_effective = data["damping_ratios_effective"]
    mode_shapes_true = data["mode_shapes_true"]
    accelerations = data["accelerations"]
    fs = data["fs"]
    T = data["T"]
    t = data["t"]
    n_samples = data["n_samples"]
    
    freq_identified = result["freq_identified"]
    damping_identified = result["damping_identified"]
    mode_shapes_identified = result["mode_shapes_identified"]
    peak_indices = result["peak_indices"]
    freqs_psd = result["freqs_psd"]
    sv1 = result["sv1"]
    sv2 = result["sv2"]
    sv1_db = result["sv1_db"]
    n_freq = result["n_freq"]
    f_max_search = result["f_max_search"]
    
    def mac_value(phi_a, phi_b):
        """Modal Assurance Criterion between two mode shape vectors."""
        num = np.abs(np.dot(phi_a, phi_b)) ** 2
        den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
        return num / den if den > 0 else 0.0
    
    n_identified = len(freq_identified)
    matched = []
    used_true = set()
    
    for m in range(n_identified):
        best_idx = -1
        best_diff = np.inf
        for t_idx in range(n_dof):
            if t_idx in used_true:
                continue
            diff = abs(freq_identified[m] - freq_true[t_idx])
            if diff < best_diff:
                best_diff = diff
                best_idx = t_idx
        if best_idx >= 0 and best_diff < 1.0:
            matched.append((m, best_idx))
            used_true.add(best_idx)
    
    freq_re_list = []
    damping_re_list = []
    mac_list = []
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for idx, (m_id, m_true) in enumerate(matched):
        f_re = abs(freq_identified[m_id] - freq_true[m_true]) / freq_true[m_true]
        freq_re_list.append(f_re)
        
        d_true = damping_ratios_effective[m_true]
        d_id = damping_identified[m_id]
        d_re = abs(d_id - d_true) / d_true if d_true > 0 else 0.0
        damping_re_list.append(d_re)
        
        phi_true = mode_shapes_true[:, m_true]
        phi_id = mode_shapes_identified[:, m_id]
        if np.dot(phi_true, phi_id) < 0:
            phi_id = -phi_id
            mode_shapes_identified[:, m_id] = phi_id
        
        mac_val = mac_value(phi_true, phi_id)
        mac_list.append(mac_val)
        
        print("  Mode {}: f_true={:.4f} Hz, f_id={:.4f} Hz, RE={:.2f}%".format(
            m_true+1, freq_true[m_true], freq_identified[m_id], f_re*100))
        print("           zeta_true={:.4f}, zeta_id={:.4f}, RE={:.1f}%".format(
            d_true, d_id, d_re*100))
        print("           MAC = {:.4f}".format(mac_val))
    
    # Full MAC matrix
    mac_matrix_full = np.zeros((n_identified, n_dof))
    for i in range(n_identified):
        for j in range(n_dof):
            mac_matrix_full[i, j] = mac_value(mode_shapes_identified[:, i],
                                              mode_shapes_true[:, j])
    
    avg_freq_re = np.mean(freq_re_list) if freq_re_list else 1.0
    avg_damping_re = np.mean(damping_re_list) if damping_re_list else 1.0
    avg_mac = np.mean(mac_list) if mac_list else 0.0
    min_mac = np.min(mac_list) if mac_list else 0.0
    
    print()
    print("Average frequency RE:  {:.2f}%".format(avg_freq_re*100))
    print("Average damping RE:    {:.1f}%".format(avg_damping_re*100))
    print("Average MAC:           {:.4f}".format(avg_mac))
    print("Min MAC:               {:.4f}".format(min_mac))
    print("Modes matched:         {}/{}".format(len(matched), n_dof))
    
    # Save metrics
    metrics = {
        "task": "pyoma2_modal",
        "inverse_problem": "Operational Modal Analysis - modal parameter identification",
        "method": "Frequency Domain Decomposition (FDD) + Enhanced FDD for damping",
        "n_dof": n_dof,
        "n_modes_identified": len(matched),
        "frequencies_true_Hz": freq_true.tolist(),
        "frequencies_identified_Hz": freq_identified.tolist(),
        "frequency_RE_per_mode": [round(x * 100, 2) for x in freq_re_list],
        "avg_frequency_RE_percent": round(avg_freq_re * 100, 2),
        "damping_true": damping_ratios_effective.tolist(),
        "damping_identified": damping_identified.tolist(),
        "damping_RE_per_mode": [round(x * 100, 1) for x in damping_re_list],
        "avg_damping_RE_percent": round(avg_damping_re * 100, 1),
        "MAC_per_mode": [round(x, 4) for x in mac_list],
        "avg_MAC": round(avg_mac, 4),
        "min_MAC": round(min_mac, 4),
    }
    
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    gt_data = {
        "frequencies_Hz": freq_true,
        "damping_ratios": damping_ratios_effective,
        "mode_shapes": mode_shapes_true,
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data, allow_pickle=True)
    
    recon_data = {
        "frequencies_Hz": freq_identified,
        "damping_ratios": damping_identified,
        "mode_shapes": mode_shapes_identified,
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data, allow_pickle=True)
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    
    # (a) Time history
    ax1 = fig.add_subplot(gs[0, 0])
    t_plot_end = min(2000, n_samples)
    ax1.plot(t[:t_plot_end], accelerations[:t_plot_end, 0],
             linewidth=0.4, color='steelblue')
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Acceleration [m/s^2]', fontsize=11)
    ax1.set_title('(a) Ambient Vibration - Channel 1', fontsize=12, fontweight='bold')
    ax1.annotate("fs = {:.0f} Hz, T = {:.0f} s\nSNR = 30 dB".format(fs, T),
                 xy=(0.98, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    
    # (b) FDD singular value spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(freqs_psd, sv1_db, linewidth=0.8, color='navy', label='1st SV')
    ax2.plot(freqs_psd, 10 * np.log10(sv2 + 1e-30), linewidth=0.6,
             color='gray', alpha=0.5, label='2nd SV')
    for m, pk in enumerate(peak_indices):
        ax2.axvline(freqs_psd[pk], color='red', linestyle='--', alpha=0.6, linewidth=0.8)
        ax2.plot(freqs_psd[pk], sv1_db[pk], 'rv', markersize=8)
        ax2.annotate("{:.2f} Hz".format(freqs_psd[pk]),
                     xy=(freqs_psd[pk], sv1_db[pk]),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=8, color='red', fontweight='bold')
    ax2.set_xlabel('Frequency [Hz]', fontsize=11)
    ax2.set_ylabel('Singular Value [dB]', fontsize=11)
    ax2.set_title('(b) Frequency Domain Decomposition', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, f_max_search)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # (c) Mode shape comparison
    ax3 = fig.add_subplot(gs[1, 0])
    dof_positions = np.arange(1, n_dof + 1)
    colors_plot = plt.cm.tab10(np.linspace(0, 0.5, min(5, len(matched))))
    
    n_plot_modes = min(5, len(matched))
    for idx in range(n_plot_modes):
        m_id, m_true = matched[idx]
        phi_true = mode_shapes_true[:, m_true]
        phi_id = mode_shapes_identified[:, m_id]
        color = colors_plot[idx]
        ax3.plot(dof_positions, phi_true, 'o-', color=color, linewidth=2,
                 markersize=8, label="Mode {} GT".format(m_true+1))
        ax3.plot(dof_positions, phi_id, 's--', color=color, linewidth=2,
                 markersize=7, alpha=0.7, label="Mode {} ID".format(m_true+1))
    
    ax3.set_xlabel('DOF Number', fontsize=11)
    ax3.set_ylabel('Normalised Amplitude', fontsize=11)
    ax3.set_title('(c) Mode Shape Comparison (GT vs Identified)',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, ncol=2, loc='best')
    ax3.set_xticks(dof_positions)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linewidth=0.5)
    
    # (d) MAC matrix
    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(mac_matrix_full, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax4.set_xlabel('True Mode', fontsize=11)
    ax4.set_ylabel('Identified Mode', fontsize=11)
    ax4.set_title('(d) MAC Matrix', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(n_dof))
    ax4.set_xticklabels([str(i+1) for i in range(n_dof)])
    ax4.set_yticks(range(n_identified))
    ax4.set_yticklabels([str(i+1) for i in range(n_identified)])
    
    for i in range(n_identified):
        for j in range(n_dof):
            val = mac_matrix_full[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax4.text(j, i, "{:.2f}".format(val), ha='center', va='center',
                     fontsize=9, fontweight='bold', color=color)
    
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('MAC', fontsize=10)
    
    fig.suptitle('Operational Modal Analysis - 5-DOF System\n'
                 'Inverse: Ambient vibration -> Modal parameters (FDD method)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print()
    print("Results saved to {}/".format(RESULTS_DIR))
    print("  metrics.json, ground_truth.npy, reconstruction.npy, reconstruction_result.png")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyoma2_modal_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    try:
        # Load outer data
        print(f"Loading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Running run_inversion with args and kwargs...")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is a chained execution pattern
        if inner_data_paths:
            # Chained execution - agent_output should be callable
            print(f"Detected chained execution pattern with {len(inner_data_paths)} inner data files")
            
            for inner_path in inner_data_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_inner_output = inner_data.get('output', None)
                
                if callable(agent_output):
                    final_result = agent_output(*inner_args, **inner_kwargs)
                else:
                    final_result = agent_output
                
                std_result = std_inner_output
        else:
            # Direct execution pattern
            print("Direct execution pattern detected")
            final_result = agent_output
            std_result = std_output
        
        # Extract input data for evaluation
        # The input data should be the first argument passed to run_inversion
        input_data = args[0] if args else kwargs.get('data', None)
        
        if input_data is None:
            print("ERROR: Could not extract input data for evaluation.")
            sys.exit(1)
        
        # Evaluate agent results
        print("\n" + "="*60)
        print("EVALUATING AGENT OUTPUT")
        print("="*60)
        metrics_agent = evaluate_results(input_data, final_result)
        
        # Evaluate standard results
        print("\n" + "="*60)
        print("EVALUATING STANDARD OUTPUT")
        print("="*60)
        metrics_std = evaluate_results(input_data, std_result)
        
        # Extract key metrics for comparison
        # Using avg_MAC as the primary metric (higher is better)
        score_agent = metrics_agent.get('avg_MAC', 0.0)
        score_std = metrics_std.get('avg_MAC', 0.0)
        
        # Also consider frequency and damping errors (lower is better)
        freq_re_agent = metrics_agent.get('avg_frequency_RE_percent', 100.0)
        freq_re_std = metrics_std.get('avg_frequency_RE_percent', 100.0)
        
        damping_re_agent = metrics_agent.get('avg_damping_RE_percent', 100.0)
        damping_re_std = metrics_std.get('avg_damping_RE_percent', 100.0)
        
        min_mac_agent = metrics_agent.get('min_MAC', 0.0)
        min_mac_std = metrics_std.get('min_MAC', 0.0)
        
        modes_matched_agent = metrics_agent.get('n_modes_identified', 0)
        modes_matched_std = metrics_std.get('n_modes_identified', 0)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Avg MAC        -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
        print(f"Min MAC        -> Agent: {min_mac_agent:.4f}, Standard: {min_mac_std:.4f}")
        print(f"Freq RE (%)    -> Agent: {freq_re_agent:.2f}, Standard: {freq_re_std:.2f}")
        print(f"Damping RE (%) -> Agent: {damping_re_agent:.1f}, Standard: {damping_re_std:.1f}")
        print(f"Modes Matched  -> Agent: {modes_matched_agent}, Standard: {modes_matched_std}")
        
        # Determine success criteria
        # For MAC: higher is better, allow 10% degradation
        # For RE: lower is better, allow 50% increase (since these can be more variable)
        
        mac_acceptable = score_agent >= score_std * 0.9  # Allow 10% degradation
        min_mac_acceptable = min_mac_agent >= min_mac_std * 0.85  # Allow 15% degradation for min
        freq_acceptable = freq_re_agent <= freq_re_std * 1.5 + 1.0  # Allow 50% increase plus 1% margin
        modes_acceptable = modes_matched_agent >= modes_matched_std - 1  # Allow 1 fewer mode
        
        print("\n" + "="*60)
        print("ACCEPTANCE CRITERIA")
        print("="*60)
        print(f"Avg MAC acceptable:     {mac_acceptable}")
        print(f"Min MAC acceptable:     {min_mac_acceptable}")
        print(f"Freq RE acceptable:     {freq_acceptable}")
        print(f"Modes matched acceptable: {modes_acceptable}")
        
        # Overall pass/fail
        overall_pass = mac_acceptable and min_mac_acceptable and freq_acceptable and modes_acceptable
        
        if overall_pass:
            print("\n✓ TEST PASSED: Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()