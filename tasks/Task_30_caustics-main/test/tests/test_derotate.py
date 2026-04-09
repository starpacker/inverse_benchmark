import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_derotate import derotate
# Import verification utility
from verification_utils import recursive_check

def test_derotate():
    """
    Test script for the 'derotate' function.
    It reconstructs input/output scenarios from serialized data files.
    """
    
    # 1. Define paths to data files
    # Note: Based on the provided data capture logic, we expect a simple function execution 
    # since 'derotate' returns a tuple, not a callable.
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_derotate.pkl']
    
    main_data_path = None
    
    for path in data_paths:
        if 'standard_data_derotate.pkl' in path:
            main_data_path = path
            break

    if not main_data_path:
        print("Error: standard_data_derotate.pkl not found in provided paths.")
        sys.exit(1)

    # 2. Load the Data
    try:
        with open(main_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {main_data_path}: {e}")
        sys.exit(1)

    # 3. Extract Inputs and Expected Outputs
    func_name = data.get('func_name', 'unknown')
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Testing function: {func_name}")

    # 4. Execute the Function
    try:
        actual_output = derotate(*args, **kwargs)
    except Exception as e:
        print("Error during function execution:")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify Results
    # Since derotate returns a tuple of tensors (vx, vy), we verify against the expected tuple.
    is_match, msg = recursive_check(expected_output, actual_output)

    if is_match:
        print("TEST PASSED: Output matches expected result.")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_derotate()