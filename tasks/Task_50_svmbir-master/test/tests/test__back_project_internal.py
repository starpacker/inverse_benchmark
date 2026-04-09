import sys
import os
import dill
import numpy as np
import scipy.ndimage
import torch
import traceback
from agent__back_project_internal import _back_project_internal
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data__back_project_internal.pkl']

    # 2. Strategy Analysis
    outer_data_path = None
    inner_data_paths = []

    for path in data_paths:
        filename = os.path.basename(path)
        if "parent_function__back_project_internal" in filename:
            inner_data_paths.append(path)
        elif "standard_data__back_project_internal.pkl" in filename:
            outer_data_path = path

    if not outer_data_path:
        print("Error: Could not find the main data file 'standard_data__back_project_internal.pkl'.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Phase
    try:
        # Scenario A: Simple Function Execution
        # We first execute the primary function.
        print("Executing _back_project_internal with outer args...")
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run the function
        primary_result = _back_project_internal(*outer_args, **outer_kwargs)

        # Scenario B Check: Is there inner data implying a closure/factory pattern?
        if inner_data_paths:
            print(f"Detected {len(inner_data_paths)} inner data files. Treating result as a Callable (Factory Pattern).")
            
            if not callable(primary_result):
                print("Error: Inner data files exist, but the primary function did not return a callable.")
                sys.exit(1)

            # Iterate through all recorded calls to the inner function
            for inner_path in inner_data_paths:
                print(f"  Testing Inner Data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')

                # Execute the closure/operator
                actual_output = primary_result(*inner_args, **inner_kwargs)

                # Verify
                passed, msg = recursive_check(expected_output, actual_output)
                if not passed:
                    print(f"FAILED on inner file {os.path.basename(inner_path)}")
                    print(msg)
                    sys.exit(1)

        else:
            # Scenario A: The primary result IS the final output
            print("No inner data files detected. Treating result as final output.")
            expected_output = outer_data.get('output')
            
            passed, msg = recursive_check(expected_output, primary_result)
            if not passed:
                print("FAILED verification against outer data output.")
                print(msg)
                sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()