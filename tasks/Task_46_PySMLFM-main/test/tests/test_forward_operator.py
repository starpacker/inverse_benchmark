import sys
import os
import dill
import numpy as np
import traceback

# Robust import for torch to prevent failures in environments without it
try:
    import torch
except ImportError:
    torch = None

# Ensure the current directory is in sys.path so we can import the agent code
sys.path.append(os.getcwd())

# Import the target function and verification utility
try:
    from agent_forward_operator import forward_operator
    from verification_utils import recursive_check
except ImportError as e:
    print(f"CRITICAL: Failed to import modules. {e}")
    sys.exit(1)

def test_forward_operator():
    # 1. DATA FILE ANALYSIS
    # The provided data path implies a direct function call (Scenario A), 
    # as there are no 'parent_function' files listed.
    data_paths = ['/data/yjh/PySMLFM-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    outer_path = None
    for p in data_paths:
        if 'standard_data_forward_operator.pkl' in p:
            outer_path = p
            break
            
    if not outer_path or not os.path.exists(outer_path):
        print(f"CRITICAL: Data file not found at {outer_path}")
        sys.exit(1)

    # 2. LOAD DATA
    print(f"Loading test data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load data via dill. {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. EXECUTION (Scenario A: Simple Function)
    print("Executing forward_operator...")
    try:
        # Direct execution since no factory pattern was detected in file list
        actual_result = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"CRITICAL: Execution of forward_operator failed. {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFICATION
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
    except Exception as e:
        print(f"CRITICAL: Verification utility crashed. {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()