import sys
import os
import dill
import numpy as np
import traceback
import torch
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger, vprint, time_sync, imshift_batch, torch_phasor

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Referee Function ---
def evaluate_results(results):
    """
    Calculates metrics and summarizes reconstruction results.
    
    Args:
        results: Dictionary containing reconstruction results from run_inversion
        
    Returns:
        dict containing evaluation metrics and summary
    """
    model = results['model']
    output_path = results['output_path']
    solver_time = results['solver_time']
    loss_iters = results.get('loss_iters', [])
    
    vprint("### Reconstruction Results Summary ###")
    vprint(f"Output path: {output_path}")
    vprint(f"Total solver time: {solver_time:.3f} sec")
    
    # Get object and probe shapes
    obj_shape = model.opt_obja.shape
    probe_shape = model.get_complex_probe_view().shape
    
    vprint(f"Object shape: {obj_shape}")
    vprint(f"Probe shape: {probe_shape}")
    
    # Extract final loss if available
    if len(loss_iters) > 0:
        final_loss = loss_iters[-1]
        # Handle case where final_loss might be a tuple or list
        if isinstance(final_loss, (tuple, list)):
            final_loss_value = final_loss[0] if len(final_loss) > 0 else None
        else:
            final_loss_value = final_loss
        
        if final_loss_value is not None:
            vprint(f"Final loss: {float(final_loss_value):.6f}")
    
    # Calculate basic statistics on reconstructed object
    with torch.no_grad():
        obj_amp = model.opt_obja.detach()
        obj_phase = model.opt_objp.detach()
        
        obj_amp_mean = obj_amp.mean().item()
        obj_amp_std = obj_amp.std().item()
        obj_phase_mean = obj_phase.mean().item()
        obj_phase_std = obj_phase.std().item()
        
        vprint(f"Object amplitude - mean: {obj_amp_mean:.4f}, std: {obj_amp_std:.4f}")
        vprint(f"Object phase - mean: {obj_phase_mean:.4f}, std: {obj_phase_std:.4f}")
    
    return {
        'output_path': output_path,
        'solver_time': solver_time,
        'obj_shape': obj_shape,
        'probe_shape': probe_shape,
        'obj_amp_mean': obj_amp_mean,
        'obj_amp_std': obj_amp_std,
        'obj_phase_mean': obj_phase_mean,
        'obj_phase_std': obj_phase_std,
        'model': model
    }


def extract_primary_metric(eval_result):
    """
    Extract a primary scalar metric from evaluation results for comparison.
    For this reconstruction task, we use final loss as the primary metric.
    Lower loss is better.
    """
    if isinstance(eval_result, dict):
        # Try to get loss-related metrics first (lower is better)
        model = eval_result.get('model')
        if model is not None:
            loss_iters = getattr(model, 'loss_iters', [])
            if len(loss_iters) > 0:
                final_loss = loss_iters[-1]
                if isinstance(final_loss, (tuple, list)):
                    return float(final_loss[0]) if len(final_loss) > 0 else None
                return float(final_loss)
        
        # Fallback to object statistics (use amplitude std as proxy for reconstruction quality)
        if 'obj_amp_std' in eval_result:
            return eval_result['obj_amp_std']
        
        # Return solver time as last resort
        if 'solver_time' in eval_result:
            return eval_result['solver_time']
    
    return None


def main():
    # Data paths provided
    data_paths = ['/home/yjh/ad_pty/code_2/run_code/std_data/standard_data_run_inversion.pkl']
    
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
        # Load primary (outer) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Args type: {type(args)}, length: {len(args) if isinstance(args, (list, tuple)) else 'N/A'}")
        print(f"Kwargs keys: {kwargs.keys() if isinstance(kwargs, dict) else 'N/A'}")
        
        # Execute the agent function
        print("\n### Running agent run_inversion ###")
        agent_output = run_inversion(*args, **kwargs)
        print("### Agent run_inversion completed ###\n")
        
        if is_chained_execution:
            # Chained execution pattern
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the operator returned by run_inversion
            print("\n### Running chained operator ###")
            final_result = agent_output(*inner_args, **inner_kwargs)
            print("### Chained operator completed ###\n")
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Evaluation phase
        print("\n### Evaluating agent results ###")
        score_agent_dict = evaluate_results(final_result)
        
        print("\n### Evaluating standard results ###")
        score_std_dict = evaluate_results(std_result)
        
        # Extract primary metrics for comparison
        # For reconstruction, we compare final loss (lower is better)
        agent_loss = None
        std_loss = None
        
        if 'model' in final_result and hasattr(final_result['model'], 'loss_iters'):
            agent_loss_iters = final_result['model'].loss_iters
            if len(agent_loss_iters) > 0:
                agent_loss = agent_loss_iters[-1]
                if isinstance(agent_loss, (tuple, list)):
                    agent_loss = float(agent_loss[0])
                else:
                    agent_loss = float(agent_loss)
        
        if 'model' in std_result and hasattr(std_result['model'], 'loss_iters'):
            std_loss_iters = std_result['model'].loss_iters
            if len(std_loss_iters) > 0:
                std_loss = std_loss_iters[-1]
                if isinstance(std_loss, (tuple, list)):
                    std_loss = float(std_loss[0])
                else:
                    std_loss = float(std_loss)
        
        print(f"\n### Comparison Results ###")
        print(f"Agent final loss: {agent_loss}")
        print(f"Standard final loss: {std_loss}")
        print(f"Agent solver time: {score_agent_dict['solver_time']:.3f} sec")
        print(f"Standard solver time: {score_std_dict['solver_time']:.3f} sec")
        print(f"Agent obj_amp_mean: {score_agent_dict['obj_amp_mean']:.6f}")
        print(f"Standard obj_amp_mean: {score_std_dict['obj_amp_mean']:.6f}")
        print(f"Agent obj_phase_std: {score_agent_dict['obj_phase_std']:.6f}")
        print(f"Standard obj_phase_std: {score_std_dict['obj_phase_std']:.6f}")
        
        # Verification logic
        # For loss, lower is better - allow 10% margin
        test_passed = True
        
        if agent_loss is not None and std_loss is not None:
            # Loss comparison: agent should not be significantly worse (higher) than standard
            margin = 0.10  # 10% margin
            if agent_loss > std_loss * (1 + margin):
                print(f"\nWARNING: Agent loss ({agent_loss:.6f}) is significantly higher than standard ({std_loss:.6f})")
                test_passed = False
            else:
                print(f"\nLoss comparison PASSED: Agent loss is within acceptable range")
        else:
            # Fallback to object statistics comparison
            print("\nNo loss values available, comparing object statistics...")
            
            # Compare object amplitude statistics
            amp_mean_diff = abs(score_agent_dict['obj_amp_mean'] - score_std_dict['obj_amp_mean'])
            amp_std_diff = abs(score_agent_dict['obj_amp_std'] - score_std_dict['obj_amp_std'])
            
            # Allow 20% relative difference
            if score_std_dict['obj_amp_mean'] != 0:
                amp_mean_rel_diff = amp_mean_diff / abs(score_std_dict['obj_amp_mean'])
                if amp_mean_rel_diff > 0.20:
                    print(f"WARNING: Object amplitude mean differs by {amp_mean_rel_diff*100:.1f}%")
                    test_passed = False
            
            if score_std_dict['obj_amp_std'] != 0:
                amp_std_rel_diff = amp_std_diff / abs(score_std_dict['obj_amp_std'])
                if amp_std_rel_diff > 0.20:
                    print(f"WARNING: Object amplitude std differs by {amp_std_rel_diff*100:.1f}%")
                    test_passed = False
        
        if test_passed:
            print("\n### TEST PASSED ###")
            sys.exit(0)
        else:
            print("\n### TEST FAILED ###")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n### ERROR during test execution ###")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()