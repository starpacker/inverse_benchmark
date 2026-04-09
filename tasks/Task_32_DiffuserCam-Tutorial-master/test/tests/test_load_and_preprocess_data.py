import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the agent code
sys.path.append(os.getcwd())

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Provided Data Paths
data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def run_test():
    print("Starting test for load_and_preprocess_data...")

    # 1. Identify Data Files
    outer_path = None
    inner_path = None

    for path in data_paths:
        if "parent_function_load_and_preprocess_data" in path:
            inner_path = path
        elif "standard_data_load_and_preprocess_data.pkl" in path:
            outer_path = path

    if not outer_path:
        print("Error: No standard data file found for load_and_preprocess_data.")
        sys.exit(1)

    try:
        # 2. Load Outer Data
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')

        print(f"Loaded outer data from {outer_path}")

        # 3. Execute Function
        # Scenario A: Simple Function (The function returns the data directly)
        # Scenario B: Factory Pattern (The function returns a callable, handled if inner_path exists)
        
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)

        if inner_path:
            # Scenario B: Factory Pattern
            print(f"Detected inner data file: {inner_path}. Treating result as operator.")
            if not callable(actual_result):
                print(f"Error: Expected callable from outer function because inner data exists, but got {type(actual_result)}.")
                sys.exit(1)
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output') # Override expected output with inner result

            # Execute the operator
            actual_result = actual_result(*inner_args, **inner_kwargs)
        
        # 4. Verify Results
        passed, msg = recursive_check(expected_output, actual_result)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()