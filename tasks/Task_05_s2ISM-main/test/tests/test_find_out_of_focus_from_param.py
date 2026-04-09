import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the current directory is in the path to import the agent
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_find_out_of_focus_from_param import find_out_of_focus_from_param
except ImportError:
    print("Error: Could not import 'find_out_of_focus_from_param' from 'agent_find_out_of_focus_from_param'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Basic fallback if verification_utils is missing
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, equal_nan=True):
                return False, f"Numpy arrays mismatch. Expected shape {expected.shape}, got {actual.shape}"
        elif isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                passed, msg = recursive_check(e, a)
                if not passed:
                    return False, f"Item {i}: {msg}"
        elif expected != actual:
            return False, f"Values mismatch: {expected} != {actual}"
        return True, ""

# Data paths provided
data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_find_out_of_focus_from_param.pkl']

def run_test():
    print("Starting test for 'find_out_of_focus_from_param'...")

    # Logic to distinguish between simple function and factory pattern
    # Based on the provided code, find_out_of_focus_from_param returns (optimal_depth, PSF) directly.
    # It is likely Scenario A (Simple Function).
    
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if "standard_data_find_out_of_focus_from_param.pkl" in path:
            outer_path = path
        elif "standard_data_parent_function_find_out_of_focus_from_param" in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: No standard data file found for 'find_out_of_focus_from_param'.")
        sys.exit(1)

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        sys.exit(1)

    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # Execution Phase
    try:
        print("Executing 'find_out_of_focus_from_param'...")
        result = find_out_of_focus_from_param(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verification Phase
    # Check if this function behaved as a factory (returning a callable) or returned data directly.
    # If inner_paths exist, it implies a factory pattern was detected during recording.
    
    if inner_paths and callable(result):
        print("Detected Factory Pattern. Testing inner function calls...")
        
        # In this scenario, 'result' is the 'operator' / 'agent'
        operator = result
        
        for inner_path in inner_paths:
            print(f"Testing inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Skipping inner file {inner_path} due to load error: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)

            try:
                actual_inner_output = operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner execution failed for {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected_inner_output, actual_inner_output)
            if not passed:
                print(f"Verification failed for inner call {inner_path}: {msg}")
                sys.exit(1)
        
        print("All inner function calls passed.")

    else:
        # Scenario A: Simple Function (Direct Result)
        print("Verifying direct output...")
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"Verification failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()