import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_hankelnd_r import hankelnd_r
from verification_utils import recursive_check

# List of data paths provided
data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_hankelnd_r.pkl']

def run_test():
    print("----------------------------------------------------------------")
    print("Running test_hankelnd_r.py")
    print("----------------------------------------------------------------")

    # 1. Identify Outer and Inner data files
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_hankelnd_r.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_hankelnd_r_' in path:
            inner_path = path

    if not outer_path:
        print("ERROR: Standard data file 'standard_data_hankelnd_r.pkl' not found in paths.")
        print(f"Available paths: {data_paths}")
        sys.exit(1)

    # 2. Load Outer Data (Main Function Inputs)
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file. {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. Execute Target Function
    print("Executing hankelnd_r with loaded arguments...")
    try:
        # Note: hankelnd_r returns a numpy array view (as_strided), not a closure/function.
        # However, we must support both direct return values and closure patterns based on the test strategy prompt.
        # Based on the provided source code, hankelnd_r returns an np.ndarray.
        actual_result = hankelnd_r(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Execution of hankelnd_r failed. {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification Strategy
    # Scenario A: The function returns a value (e.g., ndarray) directly.
    # Scenario B: The function returns a callable (Factory pattern).
    
    if callable(actual_result) and inner_path:
        # Scenario B: Factory Pattern
        print(f"Detected Factory Pattern. Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data file. {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_inner_output = inner_data.get('output', None)

        print("Executing inner callable...")
        try:
            final_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Execution of inner callable failed. {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected_result = expected_inner_output
        actual_to_check = final_result

    else:
        # Scenario A: Direct Return
        print("Detected Direct Return (not a factory or no inner data found).")
        expected_result = expected_outer_output
        actual_to_check = actual_result

    # 5. Check Results
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_to_check)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()