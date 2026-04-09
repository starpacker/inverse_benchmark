import sys
import os
import dill
import numpy as np
import traceback
from agent_cov_matrix import cov_matrix
from verification_utils import recursive_check

# Provided data paths
data_paths = ['/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_cov_matrix.pkl']

def main():
    print("Starting test_cov_matrix.py...")
    
    # Filter paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'standard_data_cov_matrix.pkl' in p:
            outer_path = p
        elif 'parent_function_cov_matrix' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_cov_matrix.pkl not found in provided paths.")
        sys.exit(1)

    # Load outer data (inputs for the main function)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # Scenario A: Simple Function Execution
    # Based on the provided code, cov_matrix returns a numpy array, not a callable.
    # So we simply run the function and compare the output.
    
    try:
        print("Executing cov_matrix with loaded arguments...")
        actual_result = cov_matrix(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution of cov_matrix failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verification
    print("Verifying results...")
    
    # If there are inner paths (Scenario B - Closure/Factory), the logic would change here.
    # However, inspection of the provided cov_matrix code shows it returns a numpy array (matrix).
    # Thus, we treat the result of the first call as the final result to check.
    
    # If the function returned a callable (Scenario B), we would have needed to execute that callable 
    # using data from inner_paths. Since inner_paths is empty and the function returns an array,
    # we proceed with direct comparison.

    passed, msg = recursive_check(expected_outer_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()