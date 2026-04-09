import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, norm

# Import the agent's function
from agent_run_inversion import run_inversion

# Inject the evaluate_results function verbatim from Reference B
def evaluate_results(M_true, M_recon, results_dir=None):
    """
    Compute metrics and generate visualization for Mueller matrix recovery.
    
    Parameters
    ----------
    M_true : ndarray, shape (4, 4)
        Ground truth Mueller matrix.
    M_recon : ndarray, shape (4, 4)
        Reconstructed Mueller matrix.
    results_dir : str or None
        Directory to save results. If None, uses './results'.
    
    Returns
    -------
    metrics : dict
        Contains PSNR_dB, RMSE, Frobenius_error, CC.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute metrics
    diff = M_true - M_recon
    
    # Frobenius norm error
    frob_err = norm(diff, 'fro')
    
    # RMSE over all 16 elements
    rmse = np.sqrt(np.mean(diff**2))
    
    # PSNR
    max_val = np.max(np.abs(M_true))
    if max_val < 1e-15:
        max_val = 1.0
    if rmse < 1e-15:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(max_val / rmse)
    
    # Element-wise Pearson correlation
    t = M_true.ravel()
    r = M_recon.ravel()
    if np.std(t) < 1e-15 or np.std(r) < 1e-15:
        cc = 1.0 if np.allclose(t, r) else 0.0
    else:
        cc = float(np.corrcoef(t, r)[0, 1])
    
    metrics = {
        'PSNR_dB': round(float(psnr), 4),
        'RMSE': round(float(rmse), 8),
        'Frobenius_error': round(float(frob_err), 8),
        'CC': round(float(cc), 6),
    }
    
    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), M_true)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), M_recon)
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Panel 1: GT Mueller matrix
    ax1 = fig.add_subplot(2, 3, 1)
    vmax = max(np.max(np.abs(M_true)), np.max(np.abs(M_recon)))
    im1 = ax1.imshow(M_true, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax1.set_title('Ground Truth Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{M_true[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_true[i,j]) > 0.5*vmax else 'black')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Panel 2: Reconstructed Mueller matrix
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(M_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax2.set_title('Reconstructed Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{M_recon[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_recon[i,j]) > 0.5*vmax else 'black')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Panel 3: Error matrix
    ax3 = fig.add_subplot(2, 3, 3)
    err_mat = np.abs(M_true - M_recon)
    im3 = ax3.imshow(err_mat, cmap='hot', aspect='equal')
    ax3.set_title('Absolute Error |GT − Recon|', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{err_mat[i,j]:.4f}', ha='center', va='center', fontsize=8,
                     color='white' if err_mat[i,j] > 0.5*np.max(err_mat) else 'black')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Panel 4: Element-wise bar comparison
    ax4 = fig.add_subplot(2, 3, 4)
    idx = np.arange(16)
    labels = [f'M[{i},{j}]' for i in range(4) for j in range(4)]
    gt_vals = M_true.ravel()
    rc_vals = M_recon.ravel()
    width = 0.35
    ax4.bar(idx - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax4.bar(idx + width/2, rc_vals, width, label='Reconstructed', color='coral', alpha=0.8)
    ax4.set_xticks(idx)
    ax4.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
    ax4.set_ylabel('Element Value')
    ax4.set_title('Element-wise Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Scatter plot GT vs Recon
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(gt_vals, rc_vals, c='teal', s=60, edgecolors='k', linewidths=0.5, zorder=3)
    lims = [min(gt_vals.min(), rc_vals.min()) - 0.1, max(gt_vals.max(), rc_vals.max()) + 0.1]
    ax5.plot(lims, lims, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax5.set_xlim(lims)
    ax5.set_ylim(lims)
    ax5.set_xlabel('Ground Truth')
    ax5.set_ylabel('Reconstructed')
    ax5.set_title(f'Correlation (CC = {metrics["CC"]:.4f})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_aspect('equal')
    
    # Panel 6: Metrics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    txt = (
        f"PSNR:  {metrics['PSNR_dB']:.2f} dB\n"
        f"RMSE:  {metrics['RMSE']:.6f}\n"
        f"Frobenius Error:  {metrics['Frobenius_error']:.6f}\n"
        f"Correlation (CC):  {metrics['CC']:.6f}\n"
        f"\n"
        f"Method: Dual Rotating Retarder\n"
        f"         Polarimetry (DRR)\n"
        f"Inverse: Least-Squares (pseudoinverse)\n"
        f"Sample: Partial polarizer + retarder"
    )
    ax6.text(0.1, 0.5, txt, transform=ax6.transAxes, fontsize=13,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('Recovery Metrics', fontsize=12, fontweight='bold')
    
    fig.suptitle('Mueller Matrix Recovery — Dual Rotating Retarder Polarimetry',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[SAVE] ground_truth.npy")
    print(f"[SAVE] reconstruction.npy")
    print(f"[SAVE] metrics.json")
    print(f"[SAVE] reconstruction_result.png")
    
    return metrics


def main():
    data_paths = ['/data/yjh/katsu_mueller_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[OK] Loaded outer data from {outer_path}")
        print(f"     Keys: {list(outer_data.keys())}")
        print(f"     func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Run agent's function
    try:
        agent_output = run_inversion(*args, **kwargs)
        print(f"[OK] Agent run_inversion executed successfully.")
        print(f"     Agent output type: {type(agent_output)}, shape: {getattr(agent_output, 'shape', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check for chained execution
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print(f"[INFO] Chained execution detected with {len(inner_paths)} inner file(s).")
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[OK] Loaded inner data from {inner_path}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                final_result = agent_output(*inner_args, **inner_kwargs)
                print(f"[OK] Inner call executed successfully.")
            except Exception as e:
                print(f"[ERROR] Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("[INFO] Direct execution pattern detected.")
        final_result = agent_output
        std_result = std_output
    
    # Now we need M_true (ground truth Mueller matrix) to evaluate.
    # The evaluate_results function expects (M_true, M_recon).
    # std_result is the standard output (ground truth reconstruction).
    # final_result is the agent's reconstruction.
    # We use std_result as M_true for comparison purposes.
    
    print(f"\n[INFO] Standard result type: {type(std_result)}, shape: {getattr(std_result, 'shape', 'N/A')}")
    print(f"[INFO] Agent result type: {type(final_result)}, shape: {getattr(final_result, 'shape', 'N/A')}")
    
    if std_result is not None:
        print(f"\n[INFO] Standard result:\n{std_result}")
    print(f"\n[INFO] Agent result:\n{final_result}")
    
    # Evaluate: Use std_result as ground truth, agent result as reconstruction
    # This measures how close the agent's output is to the standard output
    try:
        agent_metrics = evaluate_results(
            M_true=np.array(std_result),
            M_recon=np.array(final_result),
            results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_agent')
        )
        print(f"\n[METRICS] Agent vs Standard:")
        for k, v in agent_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Also evaluate std vs std as a baseline (should be perfect)
    try:
        std_metrics = evaluate_results(
            M_true=np.array(std_result),
            M_recon=np.array(std_result),
            results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_std')
        )
        print(f"\n[METRICS] Standard vs Standard (baseline):")
        for k, v in std_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"[ERROR] Standard baseline evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Verification
    # PSNR: Higher is better. CC: Higher is better (closer to 1).
    # RMSE: Lower is better. Frobenius_error: Lower is better.
    
    psnr_agent = agent_metrics['PSNR_dB']
    rmse_agent = agent_metrics['RMSE']
    cc_agent = agent_metrics['CC']
    frob_agent = agent_metrics['Frobenius_error']
    
    print(f"\n{'='*60}")
    print(f"FINAL SCORES:")
    print(f"  Agent PSNR: {psnr_agent} dB")
    print(f"  Agent RMSE: {rmse_agent}")
    print(f"  Agent CC:   {cc_agent}")
    print(f"  Agent Frob: {frob_agent}")
    print(f"{'='*60}")
    
    # Success criteria:
    # Since we're comparing agent output to standard output directly,
    # a perfect match gives PSNR=100, RMSE=0, CC=1, Frob=0.
    # We allow some tolerance for numerical differences.
    
    # Check if results are essentially identical (high PSNR, low RMSE)
    passed = True
    reasons = []
    
    # PSNR should be very high (>40 dB for near-identical results)
    if psnr_agent < 40.0:
        reasons.append(f"PSNR too low: {psnr_agent} dB (expected > 40 dB)")
        passed = False
    
    # CC should be very close to 1
    if cc_agent < 0.99:
        reasons.append(f"CC too low: {cc_agent} (expected > 0.99)")
        passed = False
    
    # RMSE should be very small
    if rmse_agent > 0.05:
        reasons.append(f"RMSE too high: {rmse_agent} (expected < 0.05)")
        passed = False
    
    if passed:
        print("\n[PASS] Agent performance matches standard output within acceptable tolerances.")
        sys.exit(0)
    else:
        print("\n[FAIL] Agent performance degraded:")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == '__main__':
    main()