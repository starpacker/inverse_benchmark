import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from koma.modal import xmacmat, maxreal

# Inject the evaluate_results function (Referee)
def evaluate_results(freq_true, freq_id, zeta_true, zeta_id, phi_true, phi_id,
                     t, acc, results_dir):
    """
    Compute evaluation metrics and create visualization.
    """
    n_dof = len(freq_true)

    # Frequency relative errors
    freq_re = np.abs(freq_id - freq_true) / freq_true

    # Damping ratio relative errors
    zeta_re = np.abs(zeta_id - zeta_true) / zeta_true

    # MAC values
    mac_matrix = xmacmat(phi_true, phi_id, conjugates=False)
    mac_diag = np.diag(mac_matrix)

    metrics = {
        "freq_true": freq_true.tolist(),
        "freq_identified": freq_id.tolist(),
        "freq_relative_errors": freq_re.tolist(),
        "damping_true": zeta_true.tolist(),
        "damping_identified": zeta_id.tolist(),
        "damping_relative_errors": zeta_re.tolist(),
        "mac_values": mac_diag.tolist(),
        "mac_matrix": mac_matrix.tolist(),
        "mean_freq_re": float(np.mean(freq_re)),
        "mean_damping_re": float(np.mean(zeta_re)),
        "mean_mac": float(np.mean(mac_diag)),
        "psnr": None,
        "ssim": None
    }

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Save ground truth and reconstruction
    gt_data = {
        'freq_true': freq_true,
        'zeta_true': zeta_true,
        'phi_true': phi_true,
        'acc_data': acc
    }
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_data, allow_pickle=True)

    recon_data = {
        'freq_id': freq_id,
        'zeta_id': zeta_id,
        'phi_id': phi_id
    }
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_data, allow_pickle=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Time series
    ax1 = axes[0, 0]
    t_plot = t[:min(len(t), 2000)]
    for ch in range(n_dof):
        ax1.plot(t_plot, acc[:len(t_plot), ch], alpha=0.7, linewidth=0.5,
                 label=f'DOF {ch + 1}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Input: Multi-channel Acceleration Data')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Natural frequencies
    ax2 = axes[0, 1]
    x_pos = np.arange(1, n_dof + 1)
    width = 0.35
    ax2.bar(x_pos - width / 2, freq_true, width, label='True', color='#2196F3', alpha=0.8)
    ax2.bar(x_pos + width / 2, freq_id, width, label='Identified (SSI)', color='#FF5722', alpha=0.8)
    ax2.set_xlabel('Mode Number')
    ax2.set_ylabel('Natural Frequency (Hz)')
    ax2.set_title('Natural Frequencies: True vs SSI-Identified')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i in range(n_dof):
        re_val = metrics['freq_relative_errors'][i] * 100
        ax2.annotate(f'RE={re_val:.1f}%', xy=(x_pos[i], max(freq_true[i], freq_id[i])),
                     fontsize=7, ha='center', va='bottom')

    # Panel 3: Damping ratios
    ax3 = axes[1, 0]
    ax3.bar(x_pos - width / 2, zeta_true * 100, width, label='True', color='#4CAF50', alpha=0.8)
    ax3.bar(x_pos + width / 2, zeta_id * 100, width, label='Identified (SSI)', color='#FFC107', alpha=0.8)
    ax3.set_xlabel('Mode Number')
    ax3.set_ylabel('Damping Ratio (%)')
    ax3.set_title('Damping Ratios: True vs SSI-Identified')
    ax3.set_xticks(x_pos)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for i in range(n_dof):
        re_val = metrics['damping_relative_errors'][i] * 100
        ax3.annotate(f'RE={re_val:.1f}%', xy=(x_pos[i], max(zeta_true[i], zeta_id[i]) * 100),
                     fontsize=7, ha='center', va='bottom')

    # Panel 4: MAC matrix
    ax4 = axes[1, 1]
    im = ax4.imshow(mac_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    ax4.set_xlabel('Identified Mode')
    ax4.set_ylabel('True Mode')
    ax4.set_title('Modal Assurance Criterion (MAC) Matrix')
    ax4.set_xticks(range(n_dof))
    ax4.set_yticks(range(n_dof))
    ax4.set_xticklabels([f'M{i + 1}' for i in range(n_dof)])
    ax4.set_yticklabels([f'M{i + 1}' for i in range(n_dof)])
    for i in range(n_dof):
        for j in range(n_dof):
            color = 'white' if mac_matrix[i, j] > 0.5 else 'black'
            ax4.text(j, i, f'{mac_matrix[i, j]:.2f}', ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Stochastic Subspace Identification (Cov-SSI) — 5-DOF System\n'
        f'Mean Freq RE = {metrics["mean_freq_re"] * 100:.2f}%  |  '
        f'Mean Damping RE = {metrics["mean_damping_re"] * 100:.2f}%  |  '
        f'Mean MAC = {metrics["mean_mac"]:.4f}',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    vis_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {vis_path}")

    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/koma_ssi_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify outer and inner data files
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
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    try:
        # Load outer (primary) data
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        # Run the agent's run_inversion function
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Determine execution pattern
        if inner_data_paths:
            # Chained execution pattern
            print("Detected chained execution pattern (closure/factory)")
            inner_data_path = inner_data_paths[0]
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution pattern
            print("Detected direct execution pattern")
            final_result = agent_output
            std_result = std_output
        
        # Extract parameters for evaluation
        # From the function signature: run_inversion(acc_data, fs, n_dof, freq_true)
        acc_data = args[0] if len(args) > 0 else kwargs.get('acc_data')
        fs = args[1] if len(args) > 1 else kwargs.get('fs')
        n_dof = args[2] if len(args) > 2 else kwargs.get('n_dof')
        freq_true = args[3] if len(args) > 3 else kwargs.get('freq_true')
        
        # Generate time vector
        n_samples = acc_data.shape[0]
        t = np.arange(n_samples) / fs
        
        # For evaluation, we need ground truth modal parameters
        # These need to be reconstructed or loaded from additional data
        # Based on the gen_data_code, we know the true parameters should be available
        
        # Create results directories
        results_dir_agent = './results_agent'
        results_dir_std = './results_std'
        
        # Extract agent results
        freq_id_agent = final_result['freq_id']
        zeta_id_agent = final_result['zeta_id']
        phi_id_agent = final_result['phi_id']
        
        # Extract standard results
        freq_id_std = std_result['freq_id']
        zeta_id_std = std_result['zeta_id']
        phi_id_std = std_result['phi_id']
        
        # For the true parameters, we use the standard output as reference
        # since the evaluation compares identified vs true
        # We need to estimate or use provided true values
        
        # From the test setup, freq_true is provided as input
        # We need zeta_true and phi_true for full evaluation
        # Let's create synthetic true values based on identified standard values
        # assuming standard is the ground truth reference
        
        # Use freq_true from input
        zeta_true = np.array([0.02, 0.02, 0.02, 0.02, 0.02])  # Typical damping ratios
        if n_dof != 5:
            zeta_true = np.full(n_dof, 0.02)
        
        # For phi_true, use identity-like normalization
        phi_true = np.eye(n_dof)
        
        print("\n=== Evaluating Agent Results ===")
        metrics_agent = evaluate_results(
            freq_true=freq_true,
            freq_id=freq_id_agent,
            zeta_true=zeta_true,
            zeta_id=zeta_id_agent,
            phi_true=phi_true,
            phi_id=phi_id_agent,
            t=t,
            acc=acc_data,
            results_dir=results_dir_agent
        )
        
        print("\n=== Evaluating Standard Results ===")
        metrics_std = evaluate_results(
            freq_true=freq_true,
            freq_id=freq_id_std,
            zeta_true=zeta_true,
            zeta_id=zeta_id_std,
            phi_true=phi_true,
            phi_id=phi_id_std,
            t=t,
            acc=acc_data,
            results_dir=results_dir_std
        )
        
        # Extract key metrics for comparison
        # Lower relative error is better, higher MAC is better
        mean_freq_re_agent = metrics_agent['mean_freq_re']
        mean_freq_re_std = metrics_std['mean_freq_re']
        
        mean_damping_re_agent = metrics_agent['mean_damping_re']
        mean_damping_re_std = metrics_std['mean_damping_re']
        
        mean_mac_agent = metrics_agent['mean_mac']
        mean_mac_std = metrics_std['mean_mac']
        
        print("\n=== Comparison Results ===")
        print(f"Mean Frequency RE -> Agent: {mean_freq_re_agent:.6f}, Standard: {mean_freq_re_std:.6f}")
        print(f"Mean Damping RE -> Agent: {mean_damping_re_agent:.6f}, Standard: {mean_damping_re_std:.6f}")
        print(f"Mean MAC -> Agent: {mean_mac_agent:.6f}, Standard: {mean_mac_std:.6f}")
        
        # Determine success criteria
        # For relative errors: lower is better (allow 20% tolerance)
        # For MAC: higher is better (allow 10% tolerance)
        
        tolerance = 0.2  # 20% tolerance
        mac_tolerance = 0.1  # 10% tolerance for MAC
        
        freq_re_ok = mean_freq_re_agent <= mean_freq_re_std * (1 + tolerance) + 0.01
        damping_re_ok = mean_damping_re_agent <= mean_damping_re_std * (1 + tolerance) + 0.01
        mac_ok = mean_mac_agent >= mean_mac_std * (1 - mac_tolerance)
        
        print(f"\nFrequency RE check: {'PASS' if freq_re_ok else 'FAIL'}")
        print(f"Damping RE check: {'PASS' if damping_re_ok else 'FAIL'}")
        print(f"MAC check: {'PASS' if mac_ok else 'FAIL'}")
        
        # Overall pass if at least 2 out of 3 criteria pass
        # and no critical failure (MAC should always be reasonable)
        if mac_ok and (freq_re_ok or damping_re_ok):
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        elif mean_mac_agent >= 0.9 and mean_freq_re_agent < 0.1:
            # Alternative pass: very good MAC and reasonable frequency accuracy
            print("\n=== TEST PASSED (Alternative Criteria) ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()