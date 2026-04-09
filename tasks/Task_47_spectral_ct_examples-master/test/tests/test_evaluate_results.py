import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'. Make sure the file exists.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_paths = []

    # 2. Categorize Paths
    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results_' in path:
            inner_paths.append(path)
    
    if not outer_path:
        print("Error: 'standard_data_evaluate_results.pkl' not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")

    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        sys.exit(1)

    # 4. Execute Function
    # Based on the function definition provided, evaluate_results takes (reconstruction, gt_images)
    # and returns None (it just prints metrics). However, the decoration logic might have captured
    # None as the output, or if the function was modified to return something.
    # We proceed assuming standard execution flow.
    
    print("Executing 'evaluate_results' with loaded arguments...")
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Check if this is a factory/closure scenario
        actual_result = evaluate_results(*args, **kwargs)
        
    except Exception as e:
        print(f"Error executing 'evaluate_results': {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Handle Results based on Scenario
    
    # Scenario B: Closure/Factory pattern (if inner files exist)
    if inner_paths and callable(actual_result):
        print("Detected Factory/Closure pattern. Executing returned operator...")
        
        agent_operator = actual_result
        
        for inner_path in inner_paths:
            print(f"Testing inner execution with: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                # Execute inner
                inner_actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify inner
                passed, msg = recursive_check(expected_inner_output, inner_actual_result)
                if not passed:
                    print(f"FAILED: Inner execution mismatch for {inner_path}")
                    print(msg)
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Error during inner execution for {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("All inner executions PASSED.")

    # Scenario A: Simple Function (Standard execution)
    else:
        print("Standard execution mode.")
        expected_output = outer_data.get('output')
        
        # NOTE: The provided function `evaluate_results` returns None (prints to stdout).
        # recursive_check handles None vs None correctly.
        
        passed, msg = recursive_check(expected_output, actual_result)
        if not passed:
            print("FAILED: Output mismatch.")
            print(f"Expected: {expected_output}")
            print(f"Actual: {actual_result}")
            print(msg)
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()