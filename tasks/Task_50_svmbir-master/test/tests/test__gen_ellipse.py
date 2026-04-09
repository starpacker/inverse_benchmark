import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent__gen_ellipse import _gen_ellipse
except ImportError as e:
    print(f"Error importing function: {e}")
    sys.exit(1)

from verification_utils import recursive_check

def main():
    # List of data paths provided
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data__gen_ellipse.pkl']
    
    # 1. Identify Data Files
    # We look for the primary data file for '_gen_ellipse'
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if path.endswith("standard_data__gen_ellipse.pkl"):
            outer_path = path
        elif "standard_data_parent_function" in path and "_gen_ellipse" in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: standard_data__gen_ellipse.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Reconstruct / Execute
    # The function _gen_ellipse appears to be a direct computation function (Scenario A),
    # returning a result (mask) immediately, rather than returning a closure/factory (Scenario B).
    # However, our robust logic handles both.
    
    print("Executing _gen_ellipse with loaded arguments...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run the function
        actual_result = _gen_ellipse(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print("Error during execution of _gen_ellipse:")
        traceback.print_exc()
        sys.exit(1)

    # 3. Verification
    # Since _gen_ellipse returns a value (mask) directly based on the provided signature, 
    # we compare this result against the output stored in outer_data.
    
    expected_result = outer_data.get('output')
    
    # If the function actually returned a callable (Factory pattern), we would check inner_paths here.
    # But based on the provided code, it returns 'mask * gray_level' which is an array.
    # We add a check for Scenario B just in case the provided code snippet was incomplete or usage differs.
    
    if callable(actual_result) and inner_paths:
        print("Detected Factory Pattern. Testing inner function execution...")
        operator = actual_result
        # Just test the first inner capture found
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            actual_result = operator(*inner_args, **inner_kwargs)
            
        except Exception as e:
             print(f"Error during execution of inner function from {inner_path}:")
             traceback.print_exc()
             sys.exit(1)

    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()