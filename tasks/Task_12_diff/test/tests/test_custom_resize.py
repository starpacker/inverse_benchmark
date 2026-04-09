import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch dependency to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_custom_resize import custom_resize
from verification_utils import recursive_check

def test_custom_resize():
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_custom_resize.pkl']
    
    # Locate the relevant data file
    outer_path = None
    for path in data_paths:
        if 'standard_data_custom_resize.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("Error: standard_data_custom_resize.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output')
        func_name = data.get('func_name', 'unknown')
        
        print(f"Testing function: {func_name}")
        
        # Execute the function
        actual_result = custom_resize(*args, **kwargs)
        
        # Scenario B check: If the result is callable (a factory), 
        # we would typically look for a secondary data file ('inner_path') to execute the closure.
        # However, custom_resize is an image processing function returning an array, 
        # so we proceed with direct verification.
        
        # Verify results
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_custom_resize()