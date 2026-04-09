import sys
import os
import dill
import numpy as np
import traceback
import math
import cv2

# --- 1. Target Import ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If not in path, try adding current directory
    sys.path.append(os.getcwd())
    try:
        from agent_run_inversion import run_inversion
    except ImportError as e:
        print(f"CRITICAL: Could not import target function. {e}")
        sys.exit(1)

# --- 2. Helper Injection (Referee) ---
def evaluate_results(points_3d: np.ndarray, expected_depth: float = 5000.0) -> float:
    """
    Calculates statistics on reconstructed point cloud and compares to theoretical model.
    Returns a score (Absolute Error in Z depth). Lower is better.
    """
    if points_3d.shape[0] == 0:
        print("EVALUATION FAILED: No 3D points reconstructed.")
        return float('inf')

    z_vals = points_3d[:, 2]
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    
    error = abs(z_mean - expected_depth)
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Reconstructed Points: {points_3d.shape[0]}")
    print(f"Mean Z Depth:         {z_mean:.4f} mm")
    print(f"Std Dev Z:            {z_std:.4f} mm")
    print(f"Theoretical Z:        {expected_depth:.4f} mm")
    print(f"Absolute Error:       {error:.4f} mm")
    
    # Tolerance check
    if error < 50.0: # Generous tolerance for discrete simulation
        print("STATUS: SUCCESS (Within tolerance)")
    else:
        print("STATUS: WARNING (High deviation)")
        
    # Calculate simple PSNR proxy (Linearity check logic from original code)
    # Since we don't have the phase map here, we check planarity of Z
    # Ideal surface is flat (Z = constant)
    mse = np.mean((z_vals - z_mean)**2)
    if mse == 0:
        psnr = 100
    else:
        max_val = np.max(z_vals)
        psnr = 20 * math.log10(max_val / math.sqrt(mse))
    
    print(f"Surface Planarity PSNR: {psnr:.2f} dB")
    print("="*30 + "\n")
    
    return error

# --- 3. Data Loading & Execution Logic ---
def load_and_run(data_paths):
    # Locate files
    outer_data_path = None
    inner_data_paths = []
    
    for p in data_paths:
        if "parent_function" in p:
            inner_data_paths.append(p)
        else:
            outer_data_path = p
            
    if not outer_data_path:
        print("No primary (outer) data found.")
        sys.exit(1)
        
    print(f"Loading Outer Data: {outer_data_path}")
    with open(outer_data_path, 'rb') as f:
        outer_data = dill.load(f)

    # Prepare Outer Arguments
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    
    # --- Pattern 1: Direct Execution ---
    if not inner_data_paths:
        print("Running Pattern: Direct Execution")
        try:
            # 1. Run Agent
            agent_result = run_inversion(*outer_args, **outer_kwargs)
            
            # 2. Get Ground Truth from pickle
            std_result = outer_data['output']
            
            # 3. Evaluate
            print("\n>>> Evaluating Agent Result:")
            score_agent = evaluate_results(agent_result)
            
            print("\n>>> Evaluating Standard Result:")
            score_std = evaluate_results(std_result)
            
            return score_agent, score_std
            
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    # --- Pattern 2: Chained Execution ---
    else:
        print("Running Pattern: Chained Execution (Factory/Closure)")
        # This branch implies run_inversion returns a callable, but based on the provided
        # target code, run_inversion returns np.ndarray directly. 
        # However, we must implement the logic in case the pickle structure implies it.
        # Given the provided function code for run_inversion, it is a direct function.
        # We will assume direct execution unless the code signature changes.
        # If inner paths exist, we treat them as separate test cases or closure calls.
        
        # NOTE: The provided 'run_inversion' source code is NOT a factory. 
        # It calculates and returns points_3d. 
        # If 'inner_data_paths' exist, they might be misidentified or from a different version.
        # We will proceed by running the outer only, as that matches the provided source code signature.
        
        print("Warning: Inner data paths detected but function appears to be direct execution.")
        # Proceed with Direct Execution logic on outer data
        try:
            agent_result = run_inversion(*outer_args, **outer_kwargs)
            std_result = outer_data['output']
            
            print("\n>>> Evaluating Agent Result:")
            score_agent = evaluate_results(agent_result)
            
            print("\n>>> Evaluating Standard Result:")
            score_std = evaluate_results(std_result)
            
            return score_agent, score_std
        except Exception:
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    # Define paths based on instruction context
    data_paths = ['/data/yjh/structured-light-python-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Check if files exist
    valid_paths = [p for p in data_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"Error: Data files not found at {data_paths}")
        # In a real environment, we might fail here, but for generation purposes we handle gracefully
        sys.exit(1)
        
    score_agent, score_std = load_and_run(valid_paths)
    
    print(f"Final Scores (Abs Error) -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    
    # Decision Logic: Lower error is better
    # We allow the agent to have slightly higher error (margin) or if both are within 'Success' tolerance (< 50.0)
    
    is_success = False
    
    # If agent is within absolute tolerance defined in evaluate_results
    if score_agent < 50.0:
        is_success = True
    # Or if agent is comparable to standard (within 10% margin or better)
    elif score_agent <= score_std * 1.10:
        is_success = True
        
    if is_success:
        print("TEST PASSED: Performance is acceptable.")
        sys.exit(0)
    else:
        print("TEST FAILED: Performance degradation detected.")
        sys.exit(1)