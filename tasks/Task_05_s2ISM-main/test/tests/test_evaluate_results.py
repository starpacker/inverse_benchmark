import sys
import os
import dill
import torch
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

    # 2. Identify Test Strategy (Factory vs Simple Function)
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 4. Execute Target Function
    try:
        print("Executing evaluate_results with outer arguments...")
        # Since evaluate_results saves a file, we might want to check file existence too,
        # but the primary check is return value consistency.
        actual_result = evaluate_results(*outer_args, **outer_kwargs)
        
        # Scenario check: Is this a factory function?
        # If inner_path exists, it implies evaluate_results returned a callable (Closure/Factory pattern).
        # If not, it's a standard function execution (Scenario A).
        
        if inner_path:
            # Scenario B: Factory Pattern
            if not callable(actual_result):
                print(f"Error: Expected evaluate_results to return a callable (factory pattern), but got {type(actual_result)}.")
                sys.exit(1)
            
            print(f"Loaded inner data from {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_final_output = inner_data.get('output', None)

            print("Executing inner operator...")
            final_result = actual_result(*inner_args, **inner_kwargs)
            
            # Compare Final Results
            passed, msg = recursive_check(expected_final_output, final_result)
        
        else:
            # Scenario A: Simple Function
            # The result of the first call is the final result
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