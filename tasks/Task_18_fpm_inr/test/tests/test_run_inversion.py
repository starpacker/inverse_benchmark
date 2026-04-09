import sys
import os
import dill
import numpy as np
import traceback
import torch

# Import the target function
from agent_run_inversion import run_inversion

# --- Injected Referee (Evaluation Logic) ---
def evaluate_results(result, vis_dir, sample, color):
    """
    Evaluate and save the reconstruction results.
    """
    amplitude = result['amplitude']
    phase = result['phase']
    model = result['model']
    final_loss = result['final_loss']
    final_psnr = result['final_psnr']
    
    # Compute statistics
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_min = np.min(amplitude)
    amp_max = np.max(amplitude)
    
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    
    metrics = {
        'final_loss': final_loss,
        'final_psnr': final_psnr,
        'amplitude_mean': amp_mean,
        'amplitude_std': amp_std,
        'amplitude_min': amp_min,
        'amplitude_max': amp_max,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'phase_min': phase_min,
        'phase_max': phase_max,
    }
    
    print("\n=== Reconstruction Results ===")
    print(f"Final Loss: {final_loss:.6e}")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Amplitude - Mean: {amp_mean:.4f}, Std: {amp_std:.4f}, Range: [{amp_min:.4f}, {amp_max:.4f}]")
    print(f"Phase - Mean: {phase_mean:.4f}, Std: {phase_std:.4f}, Range: [{phase_min:.4f}, {phase_max:.4f}]")
    
    # Save model
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{sample}_{color}.pth')
    
    tensors_to_save = []
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    torch.save(tensors_to_save, save_path)
    print(f"Model saved to {save_path}")
    
    # Save final images
    np.save(os.path.join(vis_dir, 'final_amplitude.npy'), amplitude)
    np.save(os.path.join(vis_dir, 'final_phase.npy'), phase)
    print(f"Results saved to {vis_dir}")
    
    return metrics


def main():
    # Data paths
    data_paths = ['/home/yjh/fpm_inr_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify file patterns
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_data_path}")
    with open(outer_data_path, 'rb') as f:
        outer_data = dill.load(f)
    
    # Extract inputs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args: {len(args)} positional arguments")
    print(f"Kwargs: {list(kwargs.keys())}")
    
    # Run the agent function
    print("\n=== Running Agent Function ===")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR during agent execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check execution pattern
    if inner_data_paths:
        # Chained execution pattern
        print("\n=== Chained Execution Pattern ===")
        inner_data_path = inner_data_paths[0]
        print(f"Loading inner data from: {inner_data_path}")
        with open(inner_data_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned operator
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR during inner execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("\n=== Direct Execution Pattern ===")
        final_result = agent_output
        std_result = std_output
    
    # Create output directories for evaluation
    agent_vis_dir = './vis_output/agent_eval'
    std_vis_dir = './vis_output/std_eval'
    os.makedirs(agent_vis_dir, exist_ok=True)
    os.makedirs(std_vis_dir, exist_ok=True)
    
    # Evaluate agent results
    print("\n=== Evaluating Agent Results ===")
    try:
        agent_metrics = evaluate_results(final_result, agent_vis_dir, 'test_sample_agent', 'test_color')
    except Exception as e:
        print(f"ERROR during agent evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard results
    print("\n=== Evaluating Standard Results ===")
    try:
        std_metrics = evaluate_results(std_result, std_vis_dir, 'test_sample_std', 'test_color')
    except Exception as e:
        print(f"ERROR during standard evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare results
    print("\n=== Comparison ===")
    agent_psnr = agent_metrics['final_psnr']
    std_psnr = std_metrics['final_psnr']
    agent_loss = agent_metrics['final_loss']
    std_loss = std_metrics['final_loss']
    
    print(f"Scores -> Agent PSNR: {agent_psnr:.2f}, Standard PSNR: {std_psnr:.2f}")
    print(f"Scores -> Agent Loss: {agent_loss:.6e}, Standard Loss: {std_loss:.6e}")
    
    # For optimization algorithms, use generous thresholds
    # PSNR: Higher is better - agent should be within 15% of standard
    # Loss: Lower is better - agent can be up to 50% higher (due to randomness)
    psnr_threshold = std_psnr * 0.85  # 15% margin
    loss_threshold = std_loss * 1.50  # 50% margin for loss
    
    # Also set absolute minimum thresholds for quality
    min_acceptable_psnr = 30.0  # Minimum acceptable PSNR in dB
    
    psnr_check = agent_psnr >= psnr_threshold
    loss_check = agent_loss <= loss_threshold
    absolute_quality_check = agent_psnr >= min_acceptable_psnr
    
    print(f"\nPSNR Check: Agent ({agent_psnr:.2f}) >= Threshold ({psnr_threshold:.2f}): {psnr_check}")
    print(f"Loss Check: Agent ({agent_loss:.6e}) <= Threshold ({loss_threshold:.6e}): {loss_check}")
    print(f"Absolute Quality Check: Agent PSNR ({agent_psnr:.2f}) >= Minimum ({min_acceptable_psnr}): {absolute_quality_check}")
    
    # Additional diagnostic metrics
    amp_mean_diff = abs(agent_metrics['amplitude_mean'] - std_metrics['amplitude_mean'])
    phase_mean_diff = abs(agent_metrics['phase_mean'] - std_metrics['phase_mean'])
    print(f"\nAmplitude Mean Difference: {amp_mean_diff:.6f}")
    print(f"Phase Mean Difference: {phase_mean_diff:.6f}")
    
    # Determine overall success
    # Pass if: (PSNR is acceptable AND Loss is acceptable) OR absolute quality is good
    # The key insight: for stochastic optimization, we care about achieving good results
    # not necessarily matching the exact same results
    
    if absolute_quality_check and (psnr_check or loss_check):
        print("\n=== TEST PASSED ===")
        print("Agent achieved acceptable performance.")
        sys.exit(0)
    else:
        # Provide detailed failure reason
        print("\n=== TEST FAILED ===")
        if not absolute_quality_check:
            print(f"Absolute quality too low: PSNR {agent_psnr:.2f} < {min_acceptable_psnr}")
        if not psnr_check:
            print(f"PSNR degraded: {agent_psnr:.2f} < {psnr_threshold:.2f} (85% of standard)")
        if not loss_check:
            print(f"Loss too high: {agent_loss:.6e} > {loss_threshold:.6e} (150% of standard)")
        sys.exit(1)


if __name__ == '__main__':
    main()