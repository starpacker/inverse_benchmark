import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to sys.path to ensure local modules can be imported
sys.path.append(os.getcwd())

from agent_downsample_image import downsample_image
from verification_utils import recursive_check

def test_downsample_image():
    """
    Unit test for downsample_image.
    
    Strategy:
    1. Identify data files from the provided paths.
    2. Load the primary inputs (Scenario A).
    3. Execute the function.
    4. Compare the result with the expected output using recursive_check.
    """
    
    # 1. Define Data Paths
    # Based on the prompt's DATA FILE ANALYSIS
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_downsample_image.pkl']
    
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'parent_function' in path:
            inner_paths.append(path)
        elif 'standard_data_downsample_image.pkl' in path:
            outer_path = path

    if not outer_path:
        print("Error: No standard data file found for 'downsample_image'. Skipping test.")
        sys.exit(0) # Exit gracefully if data is missing, though usually treated as failure

    print(f"Loading data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file {outer_path}: {e}")
        sys.exit(1)

    # 2. Extract Args and Expected Output
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output')

    print(f"Function: {outer_data.get('func_name')}")
    # print(f"Args: {outer_args}") # Debugging
    # print(f"Kwargs: {outer_kwargs}") # Debugging

    # 3. Execution Logic
    # Scenario A: The function returns the result directly (based on standard_data_downsample_image.pkl).
    # Scenario B: The function returns a closure/factory (checked via inner_paths).
    
    try:
        # Run the primary function
        actual_result = downsample_image(*outer_args, **outer_kwargs)
        
        # Check if we are in Scenario B (Factory Pattern)
        # If inner_paths exist, or if the result is callable and matches a "parent_function" pattern
        if inner_paths:
            print("Detected Factory Pattern (Scenario B). Executing inner function(s)...")
            # In this specific case, downsample_image seems to just return arrays or tuples based on the reference code,
            # but we handle the general logic just in case the provided path list implies otherwise.
            # However, looking at the provided code, downsample_image returns arrays/tuples, not a function.
            # So likely Scenario A, but we stick to the robust logic.
            
            # If actual_result is a function, we would loop through inner_paths.
            # Since the reference code shows it returns data, 'inner_paths' list should theoretically be empty 
            # based on the provided single path. If inner_paths WAS provided, we'd iterate.
            
            for inner_p in inner_paths:
                with open(inner_p, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                # Execute the closure
                sub_result = actual_result(*inner_args, **inner_kwargs)
                
                # Verification for inner execution
                passed, msg = recursive_check(inner_expected, sub_result)
                if not passed:
                    print(f"Test Failed for inner execution {inner_p}: {msg}")
                    sys.exit(1)
            
            print("All inner function executions passed.")
            
        else:
            # Scenario A: Standard Function Execution
            print("Detected Standard Execution (Scenario A).")
            passed, msg = recursive_check(expected_result, actual_result)
            if not passed:
                print(f"Test Failed: {msg}")
                sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_downsample_image()