import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing agent_scale.py to the system path
sys.path.append('/data/yjh/semiblindpsfdeconv-master_sandbox/run_code')

# Import the target function
from agent_scale import scale
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_scale.pkl']

def test_scale():
    print("----------------------------------------------------------------")
    print("Running test_scale.py")
    print("----------------------------------------------------------------")

    # 1. Identify Data Files
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'standard_data_scale.pkl' in p and 'parent_function' not in p:
            outer_path = p
        elif 'parent_function' in p:
            inner_path = p

    if outer_path is None:
        print("Error: standard_data_scale.pkl not found in data_paths.")
        sys.exit(1)

    # 2. Load Outer Data (for the main function call)
    try:
        print(f"Loading data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    
    # 3. Execute the function
    try:
        print("Executing scale(*args, **kwargs)...")
        # In this specific case, based on the function signature def scale(v),
        # it returns a result directly, not a closure.
        # So we expect Scenario A (Direct Result).
        
        actual_result = scale(*outer_args, **outer_kwargs)
        
        # Determine expected result
        # If there's an inner path (Scenario B - Closure), we would use that.
        # But looking at the provided code for `scale`, it returns an array, not a function.
        # So we stick to Scenario A logic where outer_data['output'] is the expected result.
        
        if inner_path:
            # This block handles the rare case where 'scale' might return a callable 
            # (though the provided source suggests otherwise, we keep the robust structure).
            print(f"Inner data found at {inner_path}. Treating result as operator.")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            if not callable(actual_result):
                 print(f"Error: Expected a callable from 'scale' to handle inner data, but got {type(actual_result)}.")
                 sys.exit(1)

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data['output']
            
            print("Executing operator with inner args...")
            actual_result = actual_result(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple function execution
            expected_result = outer_data['output']

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify Results
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_scale()