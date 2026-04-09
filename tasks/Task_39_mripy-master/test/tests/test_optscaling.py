import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
try:
    from agent_optscaling import optscaling
except ImportError:
    # If the file is not in the path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_optscaling import optscaling

from verification_utils import recursive_check

def run_test():
    # Define data paths
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_optscaling.pkl']
    
    # Identify data files
    outer_path = None
    inner_path = None

    # Helper to check if a path looks like the 'outer' function call data
    for p in data_paths:
        if 'standard_data_optscaling.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_optscaling_' in p: # Note: Logic adjusted based on decorator code provided ('parent_' prefix logic)
            inner_path = p

    if outer_path is None:
        print("Error: standard_data_optscaling.pkl not found in provided paths.")
        sys.exit(1)

    try:
        # Load the outer function data (Arguments to create the operator or run the function)
        print(f"Loading data from {outer_path}...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing optscaling with loaded arguments...")
        # Execute the function
        actual_result = optscaling(*outer_args, **outer_kwargs)

        # Check if the result is a closure/operator (Scenario B) or a direct value (Scenario A)
        if callable(actual_result) and inner_path:
            # Scenario B: optscaling returned a function, and we have inner data to test it
            print(f"optscaling returned a callable. Loading inner data from {inner_path}...")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)

            print("Executing the returned operator with inner arguments...")
            actual_final_result = actual_result(*inner_args, **inner_kwargs)
            
            # Compare final results
            is_match, msg = recursive_check(expected_inner_output, actual_final_result)
            if not is_match:
                print(f"FAILED: Inner function execution result mismatch.\n{msg}")
                sys.exit(1)
            else:
                print("TEST PASSED (Closure Execution)")
                sys.exit(0)
        
        else:
            # Scenario A: optscaling returned a value directly
            print("optscaling returned a value (not a callable, or no inner data provided). Comparing directly...")
            
            is_match, msg = recursive_check(expected_output, actual_result)
            if not is_match:
                print(f"FAILED: Output mismatch.\n{msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)

    except Exception as e:
        print(f"An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()