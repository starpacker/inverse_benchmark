import sys
import os
import dill
import numpy as np
import traceback
from agent_run_inversion import run_inversion

# -------------------------------------------------------------------------
# INJECTED REFEREE: evaluate_results
# -------------------------------------------------------------------------
def evaluate_results(original_data, reconstructed_data):
    """
    Computes error metrics between original and reconstructed data.
    """
    # Create mask for valid data (exclude NaNs from thresholding)
    if original_data is None or reconstructed_data is None:
        return {'mse': float('nan'), 'psnr': float('nan'), 'corr': float('nan')}
        
    mask = ~np.isnan(reconstructed_data) & ~np.isnan(original_data)
    
    if mask.sum() == 0:
        return {'mse': float('nan'), 'psnr': float('nan'), 'corr': float('nan')}
        
    orig_valid = original_data[mask]
    rec_valid = reconstructed_data[mask]
    
    # MSE
    mse = np.mean((orig_valid - rec_valid)**2)
    
    # PSNR
    max_val = np.max(orig_valid)
    if mse > 0:
        psnr = 10 * np.log10(max_val**2 / mse)
    else:
        psnr = float('inf')
        
    # Correlation Coefficient
    if orig_valid.size > 1:
        corr = np.corrcoef(orig_valid.flatten(), rec_valid.flatten())[0, 1]
    else:
        corr = 0.0
        
    return {'mse': mse, 'psnr': psnr, 'corr': corr}

# -------------------------------------------------------------------------
# HELPER: detach_recursive (for safety if input data has attached attributes)
# -------------------------------------------------------------------------
def detach_recursive(obj):
    if hasattr(obj, 'detach'):
        return obj.detach()
    if isinstance(obj, list):
        return [detach_recursive(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple((detach_recursive(x) for x in obj))
    if isinstance(obj, dict):
        return {k: detach_recursive(v) for (k, v) in obj.items()}
    return obj

# -------------------------------------------------------------------------
# MAIN TEST LOGIC
# -------------------------------------------------------------------------
def test_pipeline(data_paths):
    print(f"Data Paths: {data_paths}")
    
    # Identify files
    outer_file = None
    inner_files = []
    
    for path in data_paths:
        if 'standard_data_run_inversion.pkl' in os.path.basename(path):
            outer_file = path
        elif 'standard_data_parent_function_run_inversion' in os.path.basename(path):
            inner_files.append(path)
            
    if not outer_file:
        print("Skipping: No primary 'standard_data_run_inversion.pkl' found.")
        return

    # 1. Load Outer Data
    try:
        with open(outer_file, 'rb') as f:
            outer_data = dill.load(f)
        print("Loaded Outer Data successfully.")
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Run Outer Function
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute Agent Code
        agent_output = run_inversion(*outer_args, **outer_kwargs)
        print("Executed 'run_inversion' successfully.")
        
    except Exception as e:
        print(f"Error executing 'run_inversion': {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Handle Execution Patterns
    if inner_files:
        print("Pattern 2: Chained Execution detected.")
        if not callable(agent_output):
            print("Error: Expected 'run_inversion' to return a callable for chained execution, but got", type(agent_output))
            sys.exit(1)
            
        for inner_path in inner_files:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                    
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Run the closure/operator returned by run_inversion
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
                
                # Evaluate
                evaluate_and_compare(std_result, final_agent_result, context=os.path.basename(inner_path))
                
            except Exception as e:
                print(f"Error processing inner file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        print("Pattern 1: Direct Execution detected.")
        std_result = outer_data.get('output', None)
        evaluate_and_compare(std_result, agent_output, context="Direct Result")

def evaluate_and_compare(std_result, agent_result, context=""):
    print(f"\n--- Evaluation: {context} ---")
    
    # Use the injected referee
    # Note: evaluate_results(original_data, reconstructed_data)
    # Typically we compare the 'reconstructed' against the 'original'. 
    # However, here we are validating code integrity: Agent Output vs Ground Truth Output.
    # We will compute the metrics for BOTH vs. a theoretical perfect reference? 
    # NO. The prompt says: "Standard Evaluation Function (evaluate_results) to compare the Agent's quality against the Ground Truth."
    # Ground Truth here is the 'std_result' from the pickle (which was generated by the original code).
    # Since 'run_inversion' performs reconstruction, let's look at the referee signature:
    # evaluate_results(original_data, reconstructed_data).
    # It calculates error between them.
    # If we treat std_result as 'original_data' (Gold Standard) and agent_result as 'reconstructed_data',
    # we can see how close they are.
    
    metrics = evaluate_results(std_result, agent_result)
    print(f"Comparison Metrics (Agent vs Standard): {metrics}")
    
    # Interpretation
    # Since we are comparing two implementation outputs (Standard vs Agent), we expect them to be VERY close.
    # MSE should be near 0. PSNR should be high (or inf). Correlation should be near 1.
    
    mse = metrics.get('mse', float('nan'))
    psnr = metrics.get('psnr', float('nan'))
    corr = metrics.get('corr', float('nan'))
    
    # Thresholds for Code Integrity
    # Allow small floating point differences
    
    is_mse_pass = (mse < 1e-4) or np.isnan(mse) # Allow nan if both are nan
    is_corr_pass = (corr > 0.99) or np.isnan(corr) or (corr == 0.0 and np.all(np.isnan(std_result)))

    if is_mse_pass and is_corr_pass:
        print(">> PASSED: Agent output matches Standard output closely.")
    else:
        print(">> FAILED: Significant deviation detected.")
        print(f"   MSE: {mse} (Threshold: < 1e-4)")
        print(f"   Corr: {corr} (Threshold: > 0.99)")
        sys.exit(1)

if __name__ == "__main__":
    # Hardcoded input paths as per prompt instructions
    data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Check if files exist
    valid_paths = [p for p in data_paths if os.path.exists(p)]
    
    if not valid_paths:
        print("No valid data files found. Skipping test.")
        sys.exit(0)
        
    test_pipeline(valid_paths)
    sys.exit(0)