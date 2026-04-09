import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Referee (Evaluation Logic) ---
def evaluate_results(
    inversion_result: dict,
    data_dict: dict
) -> dict:
    """
    Evaluate the results of the inversion.
    
    Computes metrics and returns summary statistics.
    """
    model_image = inversion_result['model_image']
    shapelet_coeffs = inversion_result['shapelet_coeffs']
    chi2_reduced = inversion_result['chi2_reduced']
    elapsed_time = inversion_result['elapsed_time']
    n_max = inversion_result['n_max']
    beta = inversion_result['beta']
    
    image_data = data_dict['image_data']
    background_rms = data_dict['background_rms']
    
    # Compute residuals
    residuals = image_data - model_image
    
    # Compute RMS of residuals
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compute peak signal-to-noise ratio
    signal_max = np.max(np.abs(model_image))
    psnr = 20 * np.log10(signal_max / residual_rms) if residual_rms > 0 else np.inf
    
    # Number of shapelet coefficients
    num_coeffs = len(shapelet_coeffs)
    
    # Expected number of coefficients for given n_max
    expected_num_coeffs = int((n_max + 1) * (n_max + 2) / 2)
    
    # Compute coefficient statistics
    coeff_mean = np.mean(shapelet_coeffs)
    coeff_std = np.std(shapelet_coeffs)
    coeff_max = np.max(np.abs(shapelet_coeffs))
    
    # Print evaluation results
    print(f"Reconstruction completed in {elapsed_time:.4f} seconds.")
    print(f"Reduced Chi^2: {chi2_reduced:.4f}")
    print(f"Number of Shapelet coefficients: {num_coeffs}")
    print(f"Expected coefficients for n_max={n_max}: {expected_num_coeffs}")
    print(f"Residual RMS: {residual_rms:.6f}")
    print(f"Background RMS: {background_rms:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Shapelet beta (scale): {beta}")
    print(f"Coefficient mean: {coeff_mean:.6f}")
    print(f"Coefficient std: {coeff_std:.6f}")
    print(f"Coefficient max abs: {coeff_max:.6f}")
    
    return {
        'residuals': residuals,
        'residual_rms': residual_rms,
        'psnr': psnr,
        'chi2_reduced': chi2_reduced,
        'num_coeffs': num_coeffs,
        'elapsed_time': elapsed_time,
        'coeff_mean': coeff_mean,
        'coeff_std': coeff_std,
        'coeff_max': coeff_max
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/lenstronomy_shapelets_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
        
        # Also need data_dict for evaluation
        # The data_dict should contain 'image_data' and 'background_rms'
        # These might be in the outer_data or we need to construct them
        data_dict = outer_data.get('data_dict', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Args length: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Run the agent's run_inversion
        print("\n--- Running Agent's run_inversion ---")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent's run_inversion completed successfully.")
        
        if is_chained_execution:
            # Chained execution: agent_output is an operator/function
            inner_data_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # If data_dict is in inner_data, use it
            if data_dict is None:
                data_dict = inner_data.get('data_dict', None)
            
            print("\n--- Running Agent's operator (chained) ---")
            final_result = agent_output(*inner_args, **inner_kwargs)
            print("Chained execution completed successfully.")
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # If data_dict is still None, try to construct it from available data
        if data_dict is None:
            print("\nWARNING: data_dict not found in pickle. Attempting to construct from available data.")
            # Try to get image_data from the data_class argument
            if len(args) > 0:
                data_class = args[0]
                if hasattr(data_class, 'data'):
                    image_data = data_class.data
                else:
                    image_data = np.zeros_like(final_result['model_image'])
                    print("WARNING: Could not extract image_data from data_class, using zeros.")
            else:
                image_data = np.zeros_like(final_result['model_image'])
                print("WARNING: Could not determine image_data, using zeros.")
            
            # Try to get background_rms
            if len(args) > 0:
                data_class = args[0]
                if hasattr(data_class, 'background_rms'):
                    background_rms = data_class.background_rms
                elif hasattr(data_class, 'C_D'):
                    # C_D is the covariance matrix, background_rms might be sqrt of diagonal
                    background_rms = np.sqrt(np.mean(data_class.C_D))
                else:
                    background_rms = 1.0
                    print("WARNING: Could not extract background_rms, using 1.0.")
            else:
                background_rms = 1.0
            
            data_dict = {
                'image_data': image_data,
                'background_rms': background_rms
            }
        
        print("\n=== Evaluation Phase ===")
        
        # Evaluate agent's result
        print("\n--- Agent Result Evaluation ---")
        eval_agent = evaluate_results(final_result, data_dict)
        
        # Evaluate standard result
        print("\n--- Standard Result Evaluation ---")
        eval_std = evaluate_results(std_result, data_dict)
        
        # Extract primary metrics for comparison
        # Using PSNR (higher is better) and chi2_reduced (lower is better)
        agent_psnr = eval_agent['psnr']
        std_psnr = eval_std['psnr']
        
        agent_chi2 = eval_agent['chi2_reduced']
        std_chi2 = eval_std['chi2_reduced']
        
        agent_residual_rms = eval_agent['residual_rms']
        std_residual_rms = eval_std['residual_rms']
        
        print("\n=== Comparison Summary ===")
        print(f"Scores -> Agent PSNR: {agent_psnr:.4f}, Standard PSNR: {std_psnr:.4f}")
        print(f"Scores -> Agent Chi2: {agent_chi2:.4f}, Standard Chi2: {std_chi2:.4f}")
        print(f"Scores -> Agent Residual RMS: {agent_residual_rms:.6f}, Standard Residual RMS: {std_residual_rms:.6f}")
        
        # Determine success based on metrics
        # PSNR: Higher is better (allow 5% margin)
        # Chi2: Lower is better (allow 10% margin)
        
        success = True
        
        # Check PSNR (higher is better)
        if np.isfinite(agent_psnr) and np.isfinite(std_psnr):
            psnr_threshold = std_psnr * 0.95  # Allow 5% degradation
            if agent_psnr < psnr_threshold:
                print(f"FAIL: Agent PSNR ({agent_psnr:.4f}) is below threshold ({psnr_threshold:.4f})")
                success = False
            else:
                print(f"PASS: Agent PSNR ({agent_psnr:.4f}) meets threshold ({psnr_threshold:.4f})")
        
        # Check Chi2 (lower is better)
        if np.isfinite(agent_chi2) and np.isfinite(std_chi2):
            chi2_threshold = std_chi2 * 1.10  # Allow 10% increase
            if agent_chi2 > chi2_threshold:
                print(f"FAIL: Agent Chi2 ({agent_chi2:.4f}) exceeds threshold ({chi2_threshold:.4f})")
                success = False
            else:
                print(f"PASS: Agent Chi2 ({agent_chi2:.4f}) meets threshold ({chi2_threshold:.4f})")
        
        # Check Residual RMS (lower is better)
        if np.isfinite(agent_residual_rms) and np.isfinite(std_residual_rms):
            rms_threshold = std_residual_rms * 1.10  # Allow 10% increase
            if agent_residual_rms > rms_threshold:
                print(f"FAIL: Agent Residual RMS ({agent_residual_rms:.6f}) exceeds threshold ({rms_threshold:.6f})")
                success = False
            else:
                print(f"PASS: Agent Residual RMS ({agent_residual_rms:.6f}) meets threshold ({rms_threshold:.6f})")
        
        # Check that shapelet coefficients are comparable
        agent_num_coeffs = eval_agent['num_coeffs']
        std_num_coeffs = eval_std['num_coeffs']
        
        if agent_num_coeffs != std_num_coeffs:
            print(f"WARNING: Number of coefficients differ - Agent: {agent_num_coeffs}, Standard: {std_num_coeffs}")
        else:
            print(f"PASS: Number of coefficients match: {agent_num_coeffs}")
        
        # Final verdict
        if success:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n=== EXCEPTION OCCURRED ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()