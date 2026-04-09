import sys
import os
import dill
import numpy as np
import traceback
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Set seeds for reproducibility if necessary
def fix_seeds(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)

fix_seeds()

def main():
    """
    Unit test for verify_evaluate_results.
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 2. Analyze Paths to Determine Strategy
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        elif 'evaluate_results.pkl' in path:
            outer_data_path = path
            
    if not outer_data_path:
        print("Error: Standard data file for 'evaluate_results' not found.")
        sys.exit(1)

    print(f"Loading data from: {outer_data_path}")
    
    # 3. Load Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        sys.exit(1)
        
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output', None)

    # 4. Execution Strategy
    actual_result = None
    
    print("Executing 'evaluate_results'...")
    try:
        # Scenario A: Simple Function Execution
        # Based on the provided code, evaluate_results is a standard function returning a tuple (PSNR, SSIM).
        # It is NOT a factory/closure pattern in the provided source code, 
        # but we must handle the possibility if the data suggests a parent/child relationship.
        
        func_output = evaluate_results(*outer_args, **outer_kwargs)
        
        if inner_data_path:
            # Scenario B: Factory Pattern (Closure)
            print(f"Inner data found: {inner_data_path}. Treating result as callable operator.")
            if not callable(func_output):
                print(f"Error: Expected a callable due to inner data presence, but got {type(func_output)}.")
                sys.exit(1)
            
            # Load inner data to execute the closure
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output') # Update expected result to inner output
            
            # Execute the closure
            actual_result = func_output(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Direct Result
            actual_result = func_output

    except Exception as e:
        print("Error during execution:")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    
    # Special handling for float comparisons if recursive_check is too strict on exact float matches
    # The output is a tuple of floats (PSNR, SSIM).
    
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Detailed debug info
        print(f"Expected: {expected_result}")
        print(f"Actual:   {actual_result}")
        sys.exit(1)

if __name__ == "__main__":
    main()