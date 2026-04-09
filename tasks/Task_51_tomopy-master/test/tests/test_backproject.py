import sys
import os
import dill
import numpy as np
import scipy
import torch
import traceback

# Ensure we can import the function to test from the local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_backproject import backproject
except ImportError:
    print("CRITICAL: Failed to import 'backproject' from 'agent_backproject.py'")
    traceback.print_exc()
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("CRITICAL: Failed to import 'recursive_check' from 'verification_utils'")
    sys.exit(1)

def test_backproject():
    # 1. Setup Data Paths
    outer_path = '/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_backproject.pkl'
    
    if not os.path.exists(outer_path):
        print(f"SKIP: Data file not found at {outer_path}")
        sys.exit(0)

    # 2. Load Data
    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            data_pack = dill.load(f)
    except Exception as e:
        print(f"FAIL: Error loading data with dill: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data_pack.get('args', [])
    kwargs = data_pack.get('kwargs', {})
    expected_output = data_pack.get('output')
    
    # 3. Execution (Scenario A: Simple Function)
    print("Executing 'backproject'...")
    try:
        # Note: The backproject function relies on scipy.ndimage.
        # If the agent code is missing imports, this will raise a NameError/AttributeError.
        actual_output = backproject(*args, **kwargs)
    except Exception as e:
        print(f"FAIL: Runtime error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"FAIL: Error during recursive_check: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_backproject()