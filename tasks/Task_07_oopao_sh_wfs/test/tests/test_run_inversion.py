import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# --- Injected Referee (Evaluation Logic) ---
def evaluate_results(inversion_results, psf_ref, output_path='sh_explicit_results.png'):
    """
    Evaluate and visualize the AO correction results.
    
    Args:
        inversion_results: dict from run_inversion
        psf_ref: reference diffraction-limited PSF
        output_path: path to save the figure
    
    Returns:
        dict: Summary statistics
    """
    strehl_history = inversion_results['strehl_history']
    final_psf = inversion_results['final_psf']
    final_dm_commands = inversion_results['final_dm_commands']
    
    # Compute statistics
    initial_strehl = strehl_history[0] if len(strehl_history) > 0 else 0.0
    final_strehl = strehl_history[-1] if len(strehl_history) > 0 else 0.0
    mean_strehl = np.mean(strehl_history)
    max_strehl = np.max(strehl_history)
    min_strehl = np.min(strehl_history)
    
    # RMS of DM commands
    dm_rms = np.sqrt(np.mean(final_dm_commands ** 2))
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Strehl history
    axes[0].plot(np.arange(1, len(strehl_history) + 1), strehl_history, 'o-', linewidth=2, markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Strehl Ratio [%]')
    axes[0].set_title('Strehl Ratio Evolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(strehl_history) + 1])
    
    # Plot 2: Final PSF (log scale)
    psf_display = np.log10(final_psf + 1e-10)
    im1 = axes[1].imshow(psf_display, cmap='hot', origin='lower')
    axes[1].set_title(f'Final PSF (log scale)\nStrehl = {final_strehl:.2f}%')
    axes[1].set_xlabel('Pixels')
    axes[1].set_ylabel('Pixels')
    plt.colorbar(im1, ax=axes[1], label='log10(Intensity)')
    
    # Plot 3: Reference PSF (log scale)
    psf_ref_display = np.log10(psf_ref + 1e-10)
    im2 = axes[2].imshow(psf_ref_display, cmap='hot', origin='lower')
    axes[2].set_title('Reference PSF (Diffraction Limited)')
    axes[2].set_xlabel('Pixels')
    axes[2].set_ylabel('Pixels')
    plt.colorbar(im2, ax=axes[2], label='log10(Intensity)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print summary
    print("\n" + "=" * 60)
    print("                    EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Initial Strehl:     {initial_strehl:.2f}%")
    print(f"  Final Strehl:       {final_strehl:.2f}%")
    print(f"  Mean Strehl:        {mean_strehl:.2f}%")
    print(f"  Max Strehl:         {max_strehl:.2f}%")
    print(f"  Min Strehl:         {min_strehl:.2f}%")
    print(f"  DM RMS Command:     {dm_rms:.2e} m")
    print(f"  Number of Iters:    {len(strehl_history)}")
    print(f"  Output saved to:    {output_path}")
    print("=" * 60)
    
    return {
        'initial_strehl': initial_strehl,
        'final_strehl': final_strehl,
        'mean_strehl': mean_strehl,
        'max_strehl': max_strehl,
        'min_strehl': min_strehl,
        'dm_rms': dm_rms,
        'n_iterations': len(strehl_history)
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Load outer data
    if not outer_data_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    outer_data_path = outer_data_files[0]
    print(f"\nLoading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    # Extract args and kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Extract psf_ref from system_data for evaluation
    # The first argument should be system_data dict
    if len(args) > 0 and isinstance(args[0], dict):
        system_data = args[0]
        psf_ref = system_data.get('psf_ref', None)
    else:
        psf_ref = None
    
    if psf_ref is None:
        print("WARNING: psf_ref not found in system_data, evaluation may fail")
    
    # Check if this is chained execution (has inner data)
    if inner_data_files:
        # Pattern 2: Chained Execution
        print("\n=== Chained Execution Mode ===")
        
        # Run outer function to get operator
        print("Running run_inversion to get operator...")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Load inner data
        inner_data_path = inner_data_files[0]
        print(f"\nLoading inner data from: {inner_data_path}")
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Run operator with inner args
        print("Running operator with inner args...")
        try:
            final_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("\n=== Direct Execution Mode ===")
        
        print("Running run_inversion...")
        try:
            final_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        std_result = std_output
    
    print("\n=== Results ===")
    print(f"Agent result type: {type(final_result)}")
    print(f"Standard result type: {type(std_result)}")
    
    if isinstance(final_result, dict):
        print(f"Agent result keys: {final_result.keys()}")
    if isinstance(std_result, dict):
        print(f"Standard result keys: {std_result.keys()}")
    
    # Evaluate both results
    print("\n=== Evaluating Agent Result ===")
    try:
        if psf_ref is not None:
            eval_agent = evaluate_results(final_result, psf_ref, output_path='agent_results.png')
        else:
            # Try to extract psf_ref from result or use a dummy
            print("WARNING: Using dummy psf_ref for evaluation")
            if 'final_psf' in final_result:
                dummy_psf_ref = np.ones_like(final_result['final_psf'])
            else:
                dummy_psf_ref = np.ones((64, 64))
            eval_agent = evaluate_results(final_result, dummy_psf_ref, output_path='agent_results.png')
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        eval_agent = None
    
    print("\n=== Evaluating Standard Result ===")
    try:
        if psf_ref is not None:
            eval_std = evaluate_results(std_result, psf_ref, output_path='standard_results.png')
        else:
            if 'final_psf' in std_result:
                dummy_psf_ref = np.ones_like(std_result['final_psf'])
            else:
                dummy_psf_ref = np.ones((64, 64))
            eval_std = evaluate_results(std_result, dummy_psf_ref, output_path='standard_results.png')
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        eval_std = None
    
    # Compare scores
    print("\n=== Score Comparison ===")
    
    if eval_agent is None or eval_std is None:
        print("ERROR: Could not evaluate one or both results")
        sys.exit(1)
    
    # Use final_strehl as the primary metric (higher is better)
    score_agent = eval_agent['final_strehl']
    score_std = eval_std['final_strehl']
    
    print(f"Scores -> Agent: {score_agent:.2f}%, Standard: {score_std:.2f}%")
    print(f"Agent mean Strehl: {eval_agent['mean_strehl']:.2f}%")
    print(f"Standard mean Strehl: {eval_std['mean_strehl']:.2f}%")
    
    # Determine success (higher Strehl is better)
    # Allow 10% margin of error
    margin = 0.10
    threshold = score_std * (1 - margin)
    
    print(f"\nThreshold (90% of standard): {threshold:.2f}%")
    
    if score_agent >= threshold:
        print("SUCCESS: Agent performance is acceptable!")
        print(f"Agent Strehl ({score_agent:.2f}%) >= Threshold ({threshold:.2f}%)")
        sys.exit(0)
    else:
        print("FAILURE: Agent performance degraded significantly!")
        print(f"Agent Strehl ({score_agent:.2f}%) < Threshold ({threshold:.2f}%)")
        sys.exit(1)


if __name__ == '__main__':
    main()