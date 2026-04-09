import sys
import os
import dill
import numpy as np
import warnings
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent_precompute_rotation_matrices import precompute_rotation_matrices
except ImportError:
    print("Error: Could not import 'precompute_rotation_matrices' from 'agent_precompute_rotation_matrices.py'")
    sys.exit(1)

from verification_utils import recursive_check

def check_custom_output(expected, actual):
    """
    Custom verification for the dictionary returned by precompute_rotation_matrices.
    Standard recursive_check might fail on 'Ylm_rot' which is an object array of arrays.
    """
    if not isinstance(actual, dict):
        return False, f"Expected dict, got {type(actual)}"

    keys_to_check = ['fit', 'const', 'idx_m0']
    
    # Check standard numeric arrays
    for key in keys_to_check:
        if key not in actual:
            return False, f"Key '{key}' missing in output"
        
        # recursive_check handles standard numpy arrays well
        passed, msg = recursive_check(expected[key], actual[key])
        if not passed:
            return False, f"Mismatch in key '{key}': {msg}"

    # Special handling for 'Ylm_rot' (object array of arrays)
    key = 'Ylm_rot'
    if key in expected:
        if key not in actual:
            return False, f"Key '{key}' missing in actual output"
        
        exp_arr = expected[key]
        act_arr = actual[key]

        if not isinstance(act_arr, np.ndarray) or act_arr.dtype != object:
             return False, f"'{key}' should be a numpy array of objects"

        if exp_arr.shape != act_arr.shape:
            return False, f"Shape mismatch for '{key}'. Expected {exp_arr.shape}, got {act_arr.shape}"

        # Iterate through the object array elements
        for i in range(exp_arr.shape[0]):
            e_elem = exp_arr[i]
            a_elem = act_arr[i]
            
            # These elements are expected to be numeric arrays themselves
            passed, msg = recursive_check(e_elem, a_elem)
            if not passed:
                 return False, f"Mismatch in '{key}' at index {i}: {msg}"

    return True, ""

def run_test():
    # Data paths provided in the prompt
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_precompute_rotation_matrices.pkl']
    
    outer_path = None
    inner_paths = []

    # Identify outer and inner data files
    for path in data_paths:
        if "standard_data_precompute_rotation_matrices.pkl" in path:
            outer_path = path
        elif "standard_data_parent_function_precompute_rotation_matrices" in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: No outer data file found (standard_data_precompute_rotation_matrices.pkl).")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # Extract args/kwargs for the factory/outer function
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})

    print("Running precompute_rotation_matrices with outer args...")
    try:
        # Execute the main function
        actual_result = precompute_rotation_matrices(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine Scenario
    if inner_paths:
        print("Scenario B: Inner function calls detected (Factory Pattern).")
        # In this scenario, actual_result is expected to be a callable (the operator/closure)
        if not callable(actual_result):
            print(f"Error: Expected a callable closure from outer function, got {type(actual_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            print(f"Testing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner data: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')

            try:
                actual_inner_output = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected_inner_output, actual_inner_output)
            if not passed:
                print(f"TEST FAILED: Inner Output mismatch. {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

    else:
        print("Scenario A: No inner function calls detected. Verifying function output directly.")
        
        expected_output = outer_data.get('output')
        
        # Use custom check because 'Ylm_rot' is an object array which generic recursive_check might fail on
        passed, msg = check_custom_output(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()