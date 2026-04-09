import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent_gaussian_line import gaussian_line
from verification_utils import recursive_check

def test_gaussian_line():
    """
    Unit test for gaussian_line using recorded standard data.
    """
    
    # Define paths based on the provided list
    data_dir = '/data/yjh/carspy-main_sandbox/run_code/std_data'
    primary_data_path = os.path.join(data_dir, 'standard_data_gaussian_line.pkl')
    
    # 1. Load Data
    if not os.path.exists(primary_data_path):
        print(f"[ERROR] Data file not found at: {primary_data_path}")
        sys.exit(1)
        
    try:
        with open(primary_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load data file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)
    
    print(f"Loaded data for function: {data.get('func_name', 'unknown')}")
    
    # 2. Execution
    try:
        # Run the target function
        print("Executing gaussian_line...")
        actual_output = gaussian_line(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Verification
    # Logic: 
    # If the result is a callable (Scenario B - Factory), we would look for inner data files.
    # However, based on the function signature provided, gaussian_line returns a calculation (numpy array usually).
    # We will proceed with direct comparison.
    
    if callable(actual_output) and not isinstance(actual_output, (np.ndarray, torch.Tensor)):
        # This branch handles the case where the function unexpectedly returns a closure
        # In the context of the provided code, this is unlikely, but good for robustness if the pattern changes.
        print("[INFO] Result is a callable. Checking for secondary data files (Factory Pattern)...")
        
        # Look for inner data files
        inner_files = [f for f in os.listdir(data_dir) if 'standard_data_parent_gaussian_line_' in f]
        
        if not inner_files:
            print("[WARN] Result is callable but no inner execution data found. Comparing callable objects directly (likely to fail if not same instance).")
            # Fallback to direct comparison
        else:
            # We only test the first available inner execution for simplicity in this generated script
            inner_path = os.path.join(data_dir, inner_files[0])
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f_inner:
                    inner_data = dill.load(f_inner)
                    
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None) # Update expected output to the inner result
                
                # Execute the closure
                actual_output = actual_output(*inner_args, **inner_kwargs)
                
            except Exception as e:
                print(f"[ERROR] Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

    # Compare results
    passed, msg = recursive_check(expected_output, actual_output)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_gaussian_line()