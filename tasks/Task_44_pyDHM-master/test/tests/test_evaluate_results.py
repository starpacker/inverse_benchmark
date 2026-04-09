import sys
import os
import dill
import numpy as np
import traceback
import cv2

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # Identify data files
    outer_path = None
    inner_path = None

    # Logic to distinguish between simple function execution and factory pattern
    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        else:
            outer_path = path

    if not outer_path:
        print("Error: No standard data file found for the main function.")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")
    outer_data = load_data(outer_path)
    
    # --- Execution Phase ---
    try:
        # Step 1: Execute the main function with outer arguments
        # Since 'evaluate_results' returns a dictionary (metrics) directly and is not a factory returning a function,
        # we treat this as a standard function execution (Scenario A).
        print("Executing evaluate_results with loaded arguments...")
        result = evaluate_results(*outer_data['args'], **outer_data['kwargs'])
        
        # If there were an inner path (Scenario B - Factory), we would do this:
        if inner_path:
            print(f"Loading inner data from: {inner_path}")
            inner_data = load_data(inner_path)
            if callable(result):
                print("Executing inner function/operator...")
                result = result(*inner_data['args'], **inner_data['kwargs'])
                expected_result = inner_data['output']
            else:
                print("Error: Function expected to return a callable (factory pattern), but returned non-callable.")
                sys.exit(1)
        else:
            # Scenario A: The result of the first call is the final output
            expected_result = outer_data['output']

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Verification Phase ---
    print("Verifying results...")
    
    # Depending on how exact the floating point math is across environments, 
    # we might want to check scalar values in the dictionary specifically if recursive_check is too strict.
    # However, recursive_check usually handles approximate float comparison.
    
    passed, msg = recursive_check(expected_result, result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected_result}")
        print(f"Actual:   {result}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()