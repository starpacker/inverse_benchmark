import sys
import os
import dill
import numpy as np
import traceback

# Handle torch import optionally to prevent crashes in environments without it
try:
    import torch
except ImportError:
    torch = None

# Import the target function
# Adjust path if necessary to find agent_evaluate_results
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def run_test():
    print("Starting test_evaluate_results.py...")
    
    # 1. Define Data Paths
    # Based on the prompt instructions
    data_paths = [
        '/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]
    
    # 2. Analyze Paths for Strategy (Simple vs Factory)
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'standard_data_evaluate_results.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_evaluate_results_' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: No standard_data_evaluate_results.pkl found.")
        sys.exit(1)
        
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Logic
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output', None)

        print("Executing evaluate_results with outer arguments...")
        result_object = evaluate_results(*outer_args, **outer_kwargs)

        # Scenario B: Factory Pattern (If inner paths exist)
        if inner_paths:
            print(f"Factory pattern detected. Found {len(inner_paths)} inner data files.")
            
            if not callable(result_object):
                print(f"Error: Expected evaluate_results to return a callable for Factory pattern, but got {type(result_object)}")
                sys.exit(1)
                
            for i_path in inner_paths:
                print(f"Testing inner path: {i_path}")
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output', None)
                
                # Execute the closure/operator
                actual_inner_result = result_object(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not passed:
                    print(f"Inner Check Failed for {i_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner Check Passed for {i_path}")

        # Scenario A: Simple Function (No inner paths)
        else:
            print("Simple function pattern detected (no inner data files).")
            # The result of the first call IS the result to verify
            passed, msg = recursive_check(expected_outer_output, result_object)
            if not passed:
                print(f"Check Failed: {msg}")
                sys.exit(1)
            else:
                print("Check Passed.")

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()