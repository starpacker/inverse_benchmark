import sys
import os
import dill
import torch
import numpy as np
import traceback
from verification_utils import recursive_check

# Import the target function
from agent_max_likelihood_reconstruction import max_likelihood_reconstruction

# Import helpers if necessary to ensure dill loading context is valid
# (Based on the provided gen_data_code, these are standard imports, but we ensure torch/numpy are present)

def main():
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_max_likelihood_reconstruction.pkl']
    
    # 1. Identify File Roles
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_max_likelihood_reconstruction.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_max_likelihood_reconstruction_' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_max_likelihood_reconstruction.pkl not found in data_paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        sys.exit(1)

    # 2. Execution Strategy
    # Scenario A: No inner paths. The function `max_likelihood_reconstruction` runs and returns the final result.
    # Scenario B: Inner paths exist. `max_likelihood_reconstruction` returns a closure/operator.
    
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output')

    print("Executing max_likelihood_reconstruction with loaded arguments...")
    try:
        # Run the target function
        result_object = max_likelihood_reconstruction(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Verification Logic
    if inner_paths:
        # Scenario B: The result_object is likely a callable (Closure/Factory pattern)
        print("Inner data files detected. Treating result as an operator/closure.")
        
        if not callable(result_object):
             print(f"Error: Expected a callable operator because inner data files exist, but got {type(result_object)}.")
             # Fallback check just in case logic is mixed
             passed, msg = recursive_check(expected_outer_output, result_object)
             if passed:
                 print("Wait, the outer result matched the expected output directly. Ignoring inner files (false positive detection?).")
                 print("TEST PASSED")
                 sys.exit(0)
             else:
                 sys.exit(1)

        for i_path in inner_paths:
            print(f"Testing against inner data: {i_path}")
            try:
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner data file {i_path}: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')

            try:
                actual_inner_result = result_object(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner execution failed for {i_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"Verification FAILED for inner file {i_path}")
                print(msg)
                sys.exit(1)
            else:
                print(f"Verification PASSED for inner file {i_path}")

    else:
        # Scenario A: Simple execution
        print("No inner data files. Comparing direct output.")
        passed, msg = recursive_check(expected_outer_output, result_object)
        if not passed:
            print("Verification FAILED")
            print(msg)
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()