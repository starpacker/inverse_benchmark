import sys
import os
import dill
import traceback

# Handle imports robustly to prevent failures if torch/numpy are missing in the test runner
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from agent_C import C
from verification_utils import recursive_check

def run_test():
    # Defined data paths
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_C.pkl']
    
    # 1. Locate Data File
    # We are looking for the primary data file for function C
    file_path = None
    for p in data_paths:
        if p.endswith('standard_data_C.pkl'):
            file_path = p
            break
            
    if not file_path:
        print("Test Skipped: No data file found for 'C' in provided paths.")
        sys.exit(0)

    if not os.path.exists(file_path):
        print(f"Test Failed: Data file does not exist at {file_path}")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Test Failed: Could not load pickle data. Error: {e}")
        print("Note: If the data contains torch tensors, PyTorch must be installed in the environment.")
        sys.exit(1)

    # 3. Extract Inputs/Outputs
    # Based on gen_data_code, keys are: 'func_name', 'args', 'kwargs', 'output'
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    # 4. Execute Function
    print(f"Executing C with {len(args)} positional args and {len(kwargs)} keyword args...")
    try:
        actual_output = C(*args, **kwargs)
    except Exception as e:
        print(f"Test Failed: Execution of C raised an exception.")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    # The function C returns a slice (data), not a callable, so we compare directly.
    try:
        passed, msg = recursive_check(expected_output, actual_output)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: Output mismatch.\n{msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Test Failed: Error during verification check.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()