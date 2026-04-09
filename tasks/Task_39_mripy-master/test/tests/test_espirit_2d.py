import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the target module is in the path
# Assuming the file structure puts agent_espirit_2d.py in the same directory or importable path
try:
    from agent_espirit_2d import espirit_2d
except ImportError:
    # Fallback if running from a different directory structure
    sys.path.append(os.path.dirname(__file__))
    from agent_espirit_2d import espirit_2d

from verification_utils import recursive_check

def test_espirit_2d():
    """
    Unit test for espirit_2d function.
    Strategy:
    1. Load the standard data file provided in the paths.
    2. Extract args, kwargs, and expected output.
    3. Execute the function `espirit_2d` with the loaded inputs.
    4. Compare the result with the expected output using recursive_check.
    """
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_espirit_2d.pkl']
    
    # Identify the correct data file
    # In this case, we expect a direct function call pattern (Scenario A based on provided paths)
    target_path = None
    for path in data_paths:
        if 'standard_data_espirit_2d.pkl' in path:
            target_path = path
            break
            
    if not target_path:
        print(f"Error: Could not find standard_data_espirit_2d.pkl in {data_paths}")
        sys.exit(1)

    print(f"Loading data from {target_path}...")
    try:
        with open(target_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output', None)

    print(f"Executing espirit_2d with {len(args)} args and {len(kwargs)} kwargs...")
    
    try:
        # Execute the function
        actual_result = espirit_2d(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
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
    test_espirit_2d()