import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the target function is importable
try:
    from agent_crop2d import crop2d
except ImportError:
    # If the agent is not in the path, try adding the current directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from agent_crop2d import crop2d

from verification_utils import recursive_check

def test_crop2d():
    """
    Test script for crop2d function.
    Usage: python test_crop2d.py
    """
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_crop2d.pkl']

    # 1. Identify Data Files
    # Based on the provided path, we have the direct input/output data file.
    # The function crop2d returns data (a numpy array), not a callable, so this is Scenario A.
    primary_data_path = None
    
    for path in data_paths:
        if 'standard_data_crop2d.pkl' in path:
            primary_data_path = path
            break
            
    if not primary_data_path:
        print("Error: standard_data_crop2d.pkl not found in provided paths.")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(primary_data_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {primary_data_path}: {e}")
        sys.exit(1)

    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output')

    print(f"Loaded data from {primary_data_path}")
    print(f"Args type: {[type(a) for a in args]}")
    print(f"Kwargs keys: {list(kwargs.keys())}")

    # 3. Execution
    try:
        # Re-execute the function with captured inputs
        actual_output = crop2d(*args, **kwargs)
    except Exception as e:
        print(f"Error executing crop2d: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    try:
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_crop2d()