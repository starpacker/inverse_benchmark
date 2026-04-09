import sys
import os
import dill
import numpy as np
import traceback
import json
from itertools import permutations

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the working and results directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# INJECTED REFEREE CODE (from Reference B)
# ============================================================================

def align_endmembers(E_gt, E_rec, A_gt, A_rec):
    """
    Find optimal permutation to align estimated endmembers with GT.
    """
    R = E_gt.shape[1]

    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(R)):
        score = 0
        for i, j in enumerate(perm):
            cos_val = np.dot(E_gt[:, i], E_rec[:, j]) / (
                np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_rec[:, j]) + 1e-12
            )
            score += cos_val
        if score > best_score:
            best_score = score
            best_perm = perm

    perm_list = list(best_perm)
    E_aligned = E_rec[:, perm_list]
    A_aligned = A_rec[perm_list, :]
    return E_aligned, A_aligned, perm_list

def forward_operator(E, A):
    """
    Linear spectral mixing model: Y = E @ A
    
    Args:
        E: Endmember matrix (L x R) - spectral signatures
        A: Abundance matrix (R x P) - fractional abundances
        
    Returns:
        Y_pred: Predicted mixed spectra (L x P)
    """
    Y_pred = E @ A
    return Y_pred

def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction quality and save results.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        inversion_result: Dictionary from run_inversion
        
    Returns:
        dict: Final metrics
    """
    E_gt = data['E_gt']
    A_gt = data['A_gt']
    wavelengths = data['wavelengths']
    img_size = data['img_size']
    
    E_rec = inversion_result['E_rec']
    A_rec = inversion_result['A_rec']
    metrics = inversion_result['metrics']
    method = inversion_result['method']
    
    # Forward model verification
    print("\n[STAGE 2] Forward — Linear Mixing Model Y = E·A + N")
    Y_verify = forward_operator(E_gt, A_gt)
    Y_clean = data['Y_clean']
    fwd_error = np.linalg.norm(Y_clean - Y_verify) / np.linalg.norm(Y_clean)
    print(f"  Forward model verification error: {fwd_error:.2e}")
    
    # Print all metrics
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, list):
            print(f"  {k:30s} = {[f'{x:.4f}' for x in v]}")
        else:
            print(f"  {k:30s} = {v}")
    
    # Map to standard metric names
    std_metrics = {
        "PSNR": metrics["PSNR_abundance"],
        "CC": metrics["CC_abundance"],
        "RE": metrics["RE_abundance"],
        "RMSE": metrics["RMSE_abundance"],
        "mean_SAD_deg": metrics["mean_SAD_deg"],
        "method": method
    }
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), A_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), A_gt)
    
    # Visualize results
    visualize_results_internal(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics, img_size,
                               os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    
    return std_metrics

def visualize_results_internal(E_gt, E_rec, A_gt, A_rec, wavelengths, metrics, img_size, save_path):
    """Create multi-panel figure: endmember spectra + abundance maps."""
    E_al, A_al, _ = align_endmembers(E_gt, E_rec, A_gt, A_rec)

    R = E_gt.shape[1]
    fig = plt.figure(figsize=(20, 12))

    # Top row: endmember spectra
    for i in range(R):
        ax = fig.add_subplot(3, R, i + 1)
        ax.plot(wavelengths, E_gt[:, i], 'b-', lw=1.5, label='GT')
        ax.plot(wavelengths, E_al[:, i], 'r--', lw=1.5, label='Recon')
        ax.set_title(f'Endmember {i+1}\nSAD={metrics["per_endmember_SAD_deg"][i]:.2f}°')
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel('Reflectance')

    # Middle row: GT abundances
    A_gt_imgs = A_gt.reshape(R, img_size, img_size)
    A_rec_imgs = A_al.reshape(R, img_size, img_size)
    for i in range(R):
        ax = fig.add_subplot(3, R, R + i + 1)
        ax.imshow(A_gt_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'GT Abund. {i+1}')
        if i == 0:
            ax.set_ylabel('Ground Truth')

    # Bottom row: Reconstructed abundances
    for i in range(R):
        ax = fig.add_subplot(3, R, 2 * R + i + 1)
        ax.imshow(A_rec_imgs[i], cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Recon {i+1}\nCC={metrics["per_endmember_CC"][i]:.3f}')
        if i == 0:
            ax.set_ylabel('Reconstructed')

    fig.suptitle(
        f"HySUPP — Hyperspectral Unmixing\n"
        f"CC={metrics['CC_abundance']:.4f} | SAD={metrics['mean_SAD_deg']:.2f}° | "
        f"PSNR={metrics['PSNR_abundance']:.1f} dB",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# END OF INJECTED REFEREE CODE
# ============================================================================


def main():
    # Data paths provided
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    try:
        # Load the primary (outer) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"\nLoading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"\nExecuting run_inversion with loaded inputs...")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution pattern)
        if inner_data_files:
            print(f"\nDetected chained execution pattern...")
            inner_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                print("Warning: agent_output is not callable, using directly")
                final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print("\n" + "="*60)
        print("EVALUATION PHASE")
        print("="*60)
        
        # Get the input data (first argument to run_inversion)
        input_data = args[0] if args else kwargs.get('data', None)
        
        if input_data is None:
            print("ERROR: Could not extract input data for evaluation!")
            sys.exit(1)
        
        # Evaluate agent result
        print("\n--- Evaluating Agent Output ---")
        try:
            score_agent = evaluate_results(input_data, final_result)
            print(f"Agent metrics: {score_agent}")
        except Exception as e:
            print(f"Error evaluating agent output: {e}")
            traceback.print_exc()
            score_agent = None
        
        # Evaluate standard result
        print("\n--- Evaluating Standard Output ---")
        try:
            score_std = evaluate_results(input_data, std_result)
            print(f"Standard metrics: {score_std}")
        except Exception as e:
            print(f"Error evaluating standard output: {e}")
            traceback.print_exc()
            score_std = None
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        if score_agent is None or score_std is None:
            print("ERROR: Could not compute one or both scores!")
            sys.exit(1)
        
        # Extract the primary metric (CC - Correlation Coefficient)
        # Higher CC is better
        cc_agent = score_agent.get('CC', 0)
        cc_std = score_std.get('CC', 0)
        
        psnr_agent = score_agent.get('PSNR', 0)
        psnr_std = score_std.get('PSNR', 0)
        
        re_agent = score_agent.get('RE', 1)  # Lower is better
        re_std = score_std.get('RE', 1)
        
        sad_agent = score_agent.get('mean_SAD_deg', 90)  # Lower is better
        sad_std = score_std.get('mean_SAD_deg', 90)
        
        print(f"\nMetric Comparison:")
        print(f"  CC (higher is better):       Agent={cc_agent:.4f}, Standard={cc_std:.4f}")
        print(f"  PSNR (higher is better):     Agent={psnr_agent:.2f}, Standard={psnr_std:.2f}")
        print(f"  RE (lower is better):        Agent={re_agent:.4f}, Standard={re_std:.4f}")
        print(f"  SAD (lower is better):       Agent={sad_agent:.2f}°, Standard={sad_std:.2f}°")
        
        # Determine success based on primary metric (CC)
        # Allow 10% margin of error
        margin = 0.10
        
        # For CC (higher is better)
        cc_threshold = cc_std * (1 - margin)
        cc_pass = cc_agent >= cc_threshold
        
        # For PSNR (higher is better)
        psnr_threshold = psnr_std * (1 - margin) if psnr_std > 0 else psnr_std - 1
        psnr_pass = psnr_agent >= psnr_threshold
        
        # For RE (lower is better)
        re_threshold = re_std * (1 + margin)
        re_pass = re_agent <= re_threshold
        
        # For SAD (lower is better)
        sad_threshold = sad_std * (1 + margin)
        sad_pass = sad_agent <= sad_threshold
        
        print(f"\nThreshold Checks (10% margin):")
        print(f"  CC:   {'PASS' if cc_pass else 'FAIL'} (threshold={cc_threshold:.4f})")
        print(f"  PSNR: {'PASS' if psnr_pass else 'FAIL'} (threshold={psnr_threshold:.2f})")
        print(f"  RE:   {'PASS' if re_pass else 'FAIL'} (threshold={re_threshold:.4f})")
        print(f"  SAD:  {'PASS' if sad_pass else 'FAIL'} (threshold={sad_threshold:.2f}°)")
        
        # Overall pass: primary metric (CC) must pass
        overall_pass = cc_pass
        
        print("\n" + "="*60)
        if overall_pass:
            print("RESULT: PASS - Agent performance is acceptable")
            print("="*60)
            sys.exit(0)
        else:
            print("RESULT: FAIL - Agent performance degraded significantly")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()