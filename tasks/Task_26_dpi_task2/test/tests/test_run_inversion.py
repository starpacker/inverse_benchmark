import sys
import os
import dill
import numpy as np
import traceback
import torch

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_run_inversion import run_inversion

# Set default dtype to match the original code
torch.set_default_dtype(torch.float32)

# Inject the referee evaluation function (Reference B)
def evaluate_results(data, results, save_path):
    """
    Evaluate reconstruction results and save outputs.
    
    Args:
        data: Dictionary from load_and_preprocess_data
        results: Dictionary from run_inversion
        save_path: Directory to save results
        
    Returns:
        Dictionary containing:
            - mean_reconstruction: Mean of reconstructed samples
            - rmse: Root mean squared error vs ground truth
            - psnr: Peak signal-to-noise ratio
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_true = data['img_true']
    reconstructed = results['reconstructed']
    
    # Compute mean reconstruction
    mean_reconstruction = np.mean(reconstructed, axis=0)
    
    # Compute RMSE
    mse = np.mean((mean_reconstruction - img_true) ** 2)
    rmse = np.sqrt(mse)
    
    # Compute PSNR
    max_val = np.max(img_true)
    if mse > 0:
        psnr = 20 * np.log10(max_val / rmse)
    else:
        psnr = float('inf')
    
    # Save model and reconstruction
    torch.save(results['model'].state_dict(), os.path.join(save_path, 'mri_model.pth'))
    np.save(os.path.join(save_path, 'mri_reconstruction.npy'), reconstructed)
    np.save(os.path.join(save_path, 'mri_mean_reconstruction.npy'), mean_reconstruction)
    
    # Print metrics
    print(f"Reconstruction RMSE: {rmse:.6f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    print(f"Saved model to {os.path.join(save_path, 'mri_model.pth')}")
    print(f"Saved reconstruction to {os.path.join(save_path, 'mri_reconstruction.npy')}")
    
    return {
        'mean_reconstruction': mean_reconstruction,
        'rmse': rmse,
        'psnr': psnr
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load the primary (outer) data
        if not outer_data_files:
            print("ERROR: No outer data file found")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        
        # Extract args and kwargs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys() if kwargs else 'None'}")
        
        # Run the agent function
        print("\n" + "="*50)
        print("Running agent run_inversion...")
        print("="*50)
        
        agent_output = run_inversion(*args, **kwargs)
        
        print("\n" + "="*50)
        print("Agent execution completed")
        print("="*50)
        
        if is_chained:
            # Chained execution pattern
            inner_data_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator
            print("Executing returned operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Evaluation Phase
        print("\n" + "="*50)
        print("Evaluation Phase")
        print("="*50)
        
        # For run_inversion, we need the input 'data' dict to evaluate
        # The first argument should be the data dictionary
        input_data = args[0] if args else kwargs.get('data', None)
        
        if input_data is None:
            print("ERROR: Could not find input data for evaluation")
            sys.exit(1)
        
        # Create save directories
        agent_save_path = './agent_results'
        std_save_path = './std_results'
        
        # Evaluate agent results
        print("\nEvaluating agent results...")
        agent_metrics = evaluate_results(input_data, final_result, agent_save_path)
        
        # Evaluate standard results
        print("\nEvaluating standard results...")
        std_metrics = evaluate_results(input_data, std_result, std_save_path)
        
        # Extract primary metrics
        agent_psnr = agent_metrics['psnr']
        std_psnr = std_metrics['psnr']
        agent_rmse = agent_metrics['rmse']
        std_rmse = std_metrics['rmse']
        
        print("\n" + "="*50)
        print("RESULTS COMPARISON")
        print("="*50)
        print(f"Agent PSNR: {agent_psnr:.4f} dB")
        print(f"Standard PSNR: {std_psnr:.4f} dB")
        print(f"Agent RMSE: {agent_rmse:.6f}")
        print(f"Standard RMSE: {std_rmse:.6f}")
        
        # Determine success
        # For PSNR: Higher is better
        # For RMSE: Lower is better
        # Allow 10% margin of error
        
        margin = 0.10  # 10% margin
        
        # Check PSNR (higher is better)
        psnr_threshold = std_psnr * (1 - margin)
        psnr_pass = agent_psnr >= psnr_threshold
        
        # Check RMSE (lower is better)
        rmse_threshold = std_rmse * (1 + margin)
        rmse_pass = agent_rmse <= rmse_threshold
        
        print(f"\nPSNR threshold: {psnr_threshold:.4f} dB (agent must be >= this)")
        print(f"RMSE threshold: {rmse_threshold:.6f} (agent must be <= this)")
        print(f"PSNR test: {'PASS' if psnr_pass else 'FAIL'}")
        print(f"RMSE test: {'PASS' if rmse_pass else 'FAIL'}")
        
        # Overall pass if either metric passes (since reconstruction quality can vary)
        # Being more lenient: pass if PSNR is reasonable OR RMSE is reasonable
        overall_pass = psnr_pass or rmse_pass
        
        # Also check if the loss history converged
        if 'loss_history' in final_result and 'loss_history' in std_result:
            agent_final_loss = final_result['loss_history'][-1] if final_result['loss_history'] else float('inf')
            std_final_loss = std_result['loss_history'][-1] if std_result['loss_history'] else float('inf')
            print(f"\nFinal loss - Agent: {agent_final_loss:.4f}, Standard: {std_final_loss:.4f}")
            
            # Allow 20% margin on final loss
            loss_threshold = std_final_loss * 1.2
            loss_pass = agent_final_loss <= loss_threshold
            print(f"Loss threshold: {loss_threshold:.4f}")
            print(f"Loss test: {'PASS' if loss_pass else 'FAIL'}")
            
            # Include loss in overall assessment
            overall_pass = overall_pass or loss_pass
        
        print("\n" + "="*50)
        if overall_pass:
            print("OVERALL: PASS - Agent performance is acceptable")
            sys.exit(0)
        else:
            print("OVERALL: FAIL - Agent performance degraded significantly")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()