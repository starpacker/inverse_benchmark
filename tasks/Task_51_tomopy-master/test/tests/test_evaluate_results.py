import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Ensure the target function and verification utilities can be imported
try:
    from agent_evaluate_results import evaluate_results
    from verification_utils import recursive_check
except ImportError as e:
    print(f"[FATAL] Import error: {e}")
    sys.exit(1)

# Paths provided in the instructions
data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    print("----------------------------------------------------------------")
    print("Test Script: test_evaluate_results.py")
    print("Target Function: evaluate_results")
    print("----------------------------------------------------------------")

    # 1. Identify File Structure
    outer_data_path = None
    inner_data_paths = []

    for p in data_paths:
        if 'standard_data_evaluate_results.pkl' in p:
            outer_data_path = p
        elif 'parent_function_evaluate_results_' in p:
            inner_data_paths.append(p)

    if not outer_data_path:
        print("[SKIP] No standard_data_evaluate_results.pkl found. Skipping test.")
        sys.exit(0)

    print(f"Loading Outer Data from: {outer_data_path}")
    try:
        outer_data = load_pkl(outer_data_path)
    except Exception as e:
        print(f"[FATAL] Failed to load outer pickle file: {e}")
        sys.exit(1)

    # 2. Reconstruct/Execute the Function
    # The provided data analysis indicates 'evaluate_results' returns a list of strings and creates a plot,
    # it does NOT return a callable (Scenario A).
    # However, we must implement the generic logic to handle potential closure patterns just in case,
    # though the provided code for evaluate_results is clearly a direct execution function.

    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')

        print("Executing evaluate_results with loaded args/kwargs...")
        
        # Suppress plotting during test execution to avoid UI backend errors on headless systems
        plt.switch_backend('Agg') 
        
        actual_result = evaluate_results(*args, **kwargs)

        # Check if the result is a callable (Closure/Factory Pattern - Scenario B)
        # In the specific case of evaluate_results, it returns a list, so this block likely won't run inner logic.
        if callable(actual_result) and inner_data_paths:
            print("Detected Closure/Factory pattern. Executing inner function(s)...")
            
            for inner_path in inner_data_paths:
                print(f"  Loading Inner Data from: {inner_path}")
                inner_data = load_pkl(inner_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')

                # Execute the closure
                closure_result = actual_result(*inner_args, **inner_kwargs)

                # Verify Inner Result
                passed, msg = recursive_check(inner_expected, closure_result)
                if not passed:
                    print(f"[FAILURE] Inner function check failed in {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  [PASS] Inner function check passed for {os.path.basename(inner_path)}")
            
            print("All inner function tests passed.")
            sys.exit(0)

        # Scenario A: Standard Execution (Direct Return)
        # This is the expected path for evaluate_results based on the provided reference code.
        print("Standard execution completed. Verifying results...")
        
        passed, msg = recursive_check(expected_output, actual_result)
        if not passed:
            print("[FAILURE] Output mismatch.")
            print(f"Expected: {expected_output}")
            print(f"Actual:   {actual_result}")
            print(f"Details:  {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"[FATAL] Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()