import sys
import os
import dill
import numpy as np
import traceback
import torch
import matplotlib.pyplot as plt

# Import target function
from agent_run_inversion import run_inversion

# --- Injected Referee Function (from Reference B) ---
def evaluate_results(reconstruction, ground_truth, save_figures=True):
    """
    Computes MSE and PSNR, and optionally saves visualization figures.
    
    Args:
        reconstruction: 3D Tensor [nz, ny, nx] - reconstructed volume
        ground_truth: 3D Tensor [nz, ny, nx] - ground truth volume
        save_figures: bool - whether to save visualization figures
    
    Returns:
        metrics: dict containing MSE and PSNR values
    """
    # Compute MSE
    mse = torch.mean((reconstruction - ground_truth) ** 2)
    
    # Compute PSNR
    max_val = ground_truth.max()
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    mse_val = mse.item()
    psnr_val = psnr.item()
    
    print(f"Result: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB")
    
    if save_figures:
        # Max projection along Z
        recon_mip = reconstruction.max(dim=0)[0].cpu().numpy()
        gt_mip = ground_truth.max(dim=0)[0].cpu().numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt_mip, cmap='inferno')
        ax[0].set_title("Ground Truth (MIP)")
        ax[1].imshow(recon_mip, cmap='inferno')
        ax[1].set_title(f"Reconstruction (MIP)\nPSNR={psnr_val:.2f}dB")
        plt.savefig("result_comparison.png")
        plt.close()
        print("Saved result_comparison.png")
    
    return {
        'mse': mse_val,
        'psnr': psnr_val
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/flfm_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
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
        std_output = outer_data.get('output', None)
        ground_truth = outer_data.get('ground_truth', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Update device in kwargs if present
        if 'device' in kwargs:
            kwargs['device'] = device
        
        # Run the agent's function
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Pattern 2: Chained Execution
            print("Detected chained execution pattern...")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # The agent_output should be a callable (operator)
            print("Executing operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
            
            # Get ground truth from inner data if available
            if 'ground_truth' in inner_data:
                ground_truth = inner_data['ground_truth']
        else:
            # Pattern 1: Direct Execution
            print("Detected direct execution pattern...")
            final_result = agent_output
            std_result = std_output
        
        # Ensure results are tensors
        if not isinstance(final_result, torch.Tensor):
            final_result = torch.tensor(final_result)
        if not isinstance(std_result, torch.Tensor):
            std_result = torch.tensor(std_result)
        
        # Move to same device for comparison
        final_result = final_result.to(device)
        std_result = std_result.to(device)
        
        print(f"Agent result shape: {final_result.shape}")
        print(f"Standard result shape: {std_result.shape}")
        
        # Evaluation Phase
        # Use standard result as ground truth for comparison
        # (since we're comparing agent's output quality against the standard)
        
        print("\n--- Evaluating Agent's Result ---")
        score_agent = evaluate_results(final_result, std_result, save_figures=True)
        
        print("\n--- Evaluating Standard Result (self-comparison) ---")
        score_std = evaluate_results(std_result, std_result, save_figures=False)
        
        # Extract primary metrics (PSNR - higher is better)
        agent_psnr = score_agent['psnr']
        std_psnr = score_std['psnr']
        agent_mse = score_agent['mse']
        std_mse = score_std['mse']
        
        print(f"\nScores -> Agent PSNR: {agent_psnr:.2f} dB, Standard PSNR: {std_psnr:.2f} dB")
        print(f"Scores -> Agent MSE: {agent_mse:.6f}, Standard MSE: {std_mse:.6f}")
        
        # Verification
        # For PSNR: Higher is better
        # The standard result compared to itself will have infinite PSNR (or very high)
        # So we need a different approach - check if agent's result is reasonable
        
        # Alternative: Check MSE between agent and standard is small
        direct_mse = torch.mean((final_result - std_result) ** 2).item()
        print(f"Direct MSE between Agent and Standard: {direct_mse:.6f}")
        
        # Calculate relative error
        std_magnitude = torch.mean(std_result ** 2).item()
        relative_error = direct_mse / (std_magnitude + 1e-8)
        print(f"Relative Error: {relative_error:.6f}")
        
        # Success criteria:
        # 1. Relative error should be small (< 10%)
        # 2. Or PSNR between agent and standard should be high (> 20 dB)
        
        psnr_threshold = 20.0  # dB
        relative_error_threshold = 0.1  # 10%
        
        if agent_psnr >= psnr_threshold or relative_error < relative_error_threshold:
            print(f"\n✓ SUCCESS: Agent's performance is acceptable.")
            print(f"  PSNR: {agent_psnr:.2f} dB (threshold: {psnr_threshold} dB)")
            print(f"  Relative Error: {relative_error:.4f} (threshold: {relative_error_threshold})")
            sys.exit(0)
        else:
            print(f"\n✗ FAILURE: Agent's performance degraded significantly.")
            print(f"  PSNR: {agent_psnr:.2f} dB (threshold: {psnr_threshold} dB)")
            print(f"  Relative Error: {relative_error:.4f} (threshold: {relative_error_threshold})")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()