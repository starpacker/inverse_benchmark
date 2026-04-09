import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

try:
    from agent_amd_stop import amd_stop
except ImportError:
    print("Error: Could not import 'amd_stop' from 'agent_amd_stop'. Ensure the file exists.")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is missing, define a fallback or fail. 
    # Assuming it's provided in the environment.
    print("Error: Could not import 'recursive_check' from 'verification_utils'.")
    sys.exit(1)

def test_amd_stop():
    # 1. DATA FILE ANALYSIS
    # Based on the provided path and function signature, this is Scenario A (Simple Function).
    # The function amd_stop returns a tuple of values, not a callable/closure.
    data_path = '/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_amd_stop.pkl'
    
    if not os.path.exists(data_path):
        print(f"CRITICAL: Data file not found at {data_path}")
        sys.exit(1)

    # 2. LOAD DATA
    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
        print(f"Loaded data from {data_path}")
    except Exception as e:
        print(f"CRITICAL: Failed to load data with dill: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. PREPARE INPUTS
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # 4. EXECUTE FUNCTION
    print("Executing amd_stop...")
    try:
        # Scenario A: Direct execution
        actual_output = amd_stop(*args, **kwargs)
    except Exception as e:
        print(f"CRITICAL: Execution of amd_stop failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. VERIFICATION
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"CRITICAL: Verification logic failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_amd_stop()