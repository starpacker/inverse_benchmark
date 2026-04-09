import sys
import os
import dill
import numpy as np
import traceback

# Import the target function and the class that causes the type mismatch
from agent_disk_harmonic import disk_harmonic, Field

# Import verification utility
from verification_utils import recursive_check

# Hardcoded paths based on instructions
data_paths = [
    '/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_disk_harmonic.pkl'
]

def custom_field_compare(expected, actual):
    """
    Handles the comparison between a deserialized Field object (often from __main__ context)
    and the local Field object from agent_disk_harmonic.
    """
    # 1. Compare underlying numpy array data
    # We treat them as pure numpy arrays to bypass class type strictness
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    
    passed_arr, msg_arr = recursive_check(expected_arr, actual_arr)
    if not passed_arr:
        return False, f"Field data mismatch: {msg_arr}"

    # 2. Compare 'grid' attribute
    # The grid object itself might have similar serialization scope issues,
    # so we rely on recursive_check which usually handles data-class-like structures well
    # if they aren't strictly enforcing type equality for every single sub-object,
    # or if the grid is a standard structure.
    if hasattr(expected, 'grid') and hasattr(actual, 'grid'):
        # Just checking if they are structurally similar
        # If the grid comparison fails due to strict types, we might need to relax it further,
        # but usually checking the array content is the most critical part for disk_harmonic.
        passed_grid, msg_grid = recursive_check(expected.grid, actual.grid)
        if not passed_grid:
            # Fallback: if grid is just metadata, sometimes strict type check fails it.
            # We print warning but might allow if data matches.
            # However, for rigorous testing, let's assume grid must match.
            return False, f"Field grid mismatch: {msg_grid}"
    elif hasattr(expected, 'grid') != hasattr(actual, 'grid'):
        return False, "One Field has .grid and the other does not."

    return True, "Field comparison passed (Custom Logic)"

def run_test():
    outer_path = None
    inner_path = None

    # Filter paths
    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        elif 'disk_harmonic.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 1. Reconstruct Operator / Result
    print("Executing disk_harmonic with outer args...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Determine if disk_harmonic is expected to return a value or a function (Closure)
        # Based on the code provided: disk_harmonic(n, m, ...) returns a Field object directly.
        # It is NOT a factory function returning a callable.
        # So 'agent_result' is the final result.
        agent_result = disk_harmonic(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Verify
    # Since disk_harmonic returns the result directly, we compare agent_result vs outer_data['output']
    expected_result = outer_data['output']

    print("Verifying result...")
    
    # Check if we are dealing with the Field class mismatch
    # We check if names match 'Field', avoiding strict class object comparison
    is_expected_field = type(expected_result).__name__ == 'Field'
    is_actual_field = type(agent_result).__name__ == 'Field'

    if is_expected_field and is_actual_field:
        passed, msg = custom_field_compare(expected_result, agent_result)
    else:
        # Standard check
        passed, msg = recursive_check(expected_result, agent_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"Test Failed: {msg}")
        # Debugging info
        print(f"Expected Type: {type(expected_result)}")
        print(f"Actual Type: {type(agent_result)}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()