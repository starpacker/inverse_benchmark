import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch import
try:
    import torch
except ImportError:
    torch = None

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def run_test():
    # 1. Define paths
    data_paths = ['/data/yjh/storm-analysis-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_path = None

    # Logic to distinguish between simple function calls and factory/closure patterns
    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        elif 'evaluate_results.pkl' in path:
            outer_path = path

    if not outer_path:
        print("Test Skipped: No outer data file found.")
        sys.exit(0)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Execute Outer Function
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the function under test
        actual_result = evaluate_results(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution Error in evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Handle Scenario A vs B
    expected_result = None
    
    if inner_path:
        # Scenario B: Factory Pattern (The result of outer is a function we must call)
        print(f"Scenario B detected. Loading inner data from: {inner_path}")
        
        if not callable(actual_result):
            print(f"Error: Expected evaluate_results to return a callable for inner execution, but got {type(actual_result)}")
            sys.exit(1)
            
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            # Execute the inner operator
            final_actual = actual_result(*inner_args, **inner_kwargs)
            expected_result = inner_data.get('output')
            actual_result = final_actual # Update for comparison
            
        except Exception as e:
            print(f"Error executing inner function from factory: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function (The result of outer is the final value)
        print("Scenario A detected: Direct function execution.")
        expected_result = outer_data.get('output')

    # 4. Verification
    try:
        is_correct, msg = recursive_check(expected_result, actual_result)
        
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_result}")
            print(f"Actual:   {actual_result}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Verification Logic Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()