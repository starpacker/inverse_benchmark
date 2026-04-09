import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
# Import verification utility
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/structured-light-python-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 2. Identify Test Strategy
    # The target function `load_and_preprocess_data` returns a dictionary directly, not a function.
    # Therefore, this is Scenario A: Simple Function Execution.
    # We only expect the standard data file, no parent_function files.
    
    outer_path = None
    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
            break
            
    if outer_path is None:
        print("Error: Standard data file not found in provided paths.")
        sys.exit(1)
        
    print(f"Loading data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution
    print("Executing load_and_preprocess_data...")
    try:
        # Extract inputs and expected output
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_result = outer_data.get('output')
        
        # Run the function
        actual_result = load_and_preprocess_data(*args, **kwargs)
        
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()