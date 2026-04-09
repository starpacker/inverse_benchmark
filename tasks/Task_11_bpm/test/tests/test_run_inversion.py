import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# --- Injected Referee (Evaluation Logic) ---
def evaluate_results(delta_ri, loss_history, ROI):
    """
    Evaluate and report reconstruction results.
    
    Args:
        delta_ri: Reconstructed refractive index (numpy array)
        loss_history: List of loss values during optimization
        ROI: Region of interest for evaluation (tuple of 6 integers)
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    s0, e0, s1, e1, s2, e2 = ROI
    
    roi_data = delta_ri[s0:e0, s1:e1, s2:e2]
    
    vmin = float(np.min(roi_data))
    vmax = float(np.max(roi_data))
    vmean = float(np.mean(roi_data))
    vstd = float(np.std(roi_data))
    
    initial_loss = loss_history[0] if len(loss_history) > 0 else None
    final_loss = loss_history[-1] if len(loss_history) > 0 else None
    
    metrics = {
        'roi_min': vmin,
        'roi_max': vmax,
        'roi_mean': vmean,
        'roi_std': vstd,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'num_iterations': len(loss_history),
        'loss_history': loss_history
    }
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROI Statistics:")
    print(f"  Min: {vmin:.6f}")
    print(f"  Max: {vmax:.6f}")
    print(f"  Mean: {vmean:.6f}")
    print(f"  Std: {vstd:.6f}")
    print(f"Optimization:")
    if initial_loss is not None:
        print(f"  Initial Loss: {initial_loss:.6f}")
    else:
        print("  Initial Loss: N/A")
    if final_loss is not None:
        print(f"  Final Loss: {final_loss:.6f}")
    else:
        print("  Final Loss: N/A")
    print(f"  Iterations: {len(loss_history)}")
    print("=" * 60)
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained_execution = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if len(outer_data_files) == 0:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys() if kwargs else 'None'}")
        
        # Run the target function
        print("Running run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained_execution:
            # Pattern 2: Chained Execution
            print("Detected chained execution pattern.")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # agent_output should be a callable (operator)
            print("Executing operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Pattern 1: Direct Execution
            print("Detected direct execution pattern.")
            final_result = agent_output
            std_result = std_output
        
        # Extract delta_ri and loss_history from results
        # run_inversion returns (delta_ri, loss_history)
        if isinstance(final_result, tuple) and len(final_result) == 2:
            agent_delta_ri, agent_loss_history = final_result
        else:
            print("ERROR: Unexpected agent output format.")
            sys.exit(1)
        
        if isinstance(std_result, tuple) and len(std_result) == 2:
            std_delta_ri, std_loss_history = std_result
        else:
            print("ERROR: Unexpected standard output format.")
            sys.exit(1)
        
        # Extract ROI from data_config (third argument)
        # args = (preprocessed_data, reconstruction_config, data_config)
        data_config = args[2] if len(args) > 2 else kwargs.get('data_config', {})
        preprocessed_data = args[0] if len(args) > 0 else kwargs.get('preprocessed_data', {})
        
        roi_config = data_config.get('ROI', [None, None, None, None, None, None])
        
        # Get domain dimensions for ROI defaults
        domain_size = preprocessed_data.get('domain_size', (100, 100, 100))
        region_z = preprocessed_data.get('region_z', domain_size[0])
        
        s0 = roi_config[0] if roi_config[0] is not None else 0
        e0 = roi_config[1] if roi_config[1] is not None else region_z
        s1 = roi_config[2] if roi_config[2] is not None else domain_size[1]
        e1 = roi_config[3] if roi_config[3] is not None else domain_size[1]
        s2 = roi_config[4] if roi_config[4] is not None else domain_size[2]
        e2 = roi_config[5] if roi_config[5] is not None else domain_size[2]
        ROI = (s0, e0, s1, e1, s2, e2)
        
        print(f"ROI: {ROI}")
        
        # Evaluation Phase
        print("\n--- Evaluating Agent Results ---")
        metrics_agent = evaluate_results(agent_delta_ri, agent_loss_history, ROI)
        
        print("\n--- Evaluating Standard Results ---")
        metrics_std = evaluate_results(std_delta_ri, std_loss_history, ROI)
        
        # Extract primary metric for comparison (final_loss - lower is better)
        score_agent = metrics_agent['final_loss']
        score_std = metrics_std['final_loss']
        
        print(f"\nScores -> Agent: {score_agent}, Standard: {score_std}")
        
        # Verification
        # For loss, lower is better
        # Allow 10% margin of error (agent can be up to 10% worse)
        if score_agent is None or score_std is None:
            print("WARNING: Could not extract loss values for comparison.")
            # Fall back to comparing ROI statistics
            agent_mean = metrics_agent['roi_mean']
            std_mean = metrics_std['roi_mean']
            agent_std = metrics_agent['roi_std']
            std_std = metrics_std['roi_std']
            
            print(f"Comparing ROI statistics instead:")
            print(f"  Agent mean: {agent_mean}, Std mean: {std_mean}")
            print(f"  Agent std: {agent_std}, Std std: {std_std}")
            
            # Check if results are reasonably close
            mean_diff = abs(agent_mean - std_mean)
            std_diff = abs(agent_std - std_std)
            
            if mean_diff < 0.1 * max(abs(std_mean), 1e-6) and std_diff < 0.1 * max(abs(std_std), 1e-6):
                print("SUCCESS: Agent results are within acceptable range.")
                sys.exit(0)
            else:
                print("FAILURE: Agent results deviate significantly from standard.")
                sys.exit(1)
        
        # For loss metric, lower is better
        # Agent is acceptable if its loss is not more than 10% higher than standard
        tolerance = 1.10  # 10% tolerance
        
        if score_agent <= score_std * tolerance:
            print(f"SUCCESS: Agent loss ({score_agent:.6f}) is within acceptable range of standard ({score_std:.6f}).")
            sys.exit(0)
        else:
            print(f"FAILURE: Agent loss ({score_agent:.6f}) exceeds acceptable threshold ({score_std * tolerance:.6f}).")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Exception occurred during testing.")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()