import sys
import os
import dill
import torch
import numpy as np
import traceback
import math

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Fix random seeds for reproducibility
def _fix_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_fix_seeds()

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/structured-light-python-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 2. Strategy Determination
    # We look for a file that represents the direct inputs to the function
    outer_data_path = None
    inner_data_paths = []

    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_data_path = path
        elif 'standard_data_parent_function_evaluate_results_' in path:
            inner_data_paths.append(path)

    if not outer_data_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Execution Logic
    try:
        # Load outer data (inputs for evaluate_results)
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_result = outer_data.get('output')

        print(f"Running evaluate_results with args len={len(outer_args)} kwargs keys={list(outer_kwargs.keys())}...")
        
        # Execute the function
        actual_result = evaluate_results(*outer_args, **outer_kwargs)

        # 4. Check if the result is a closure/operator (Scenario B) or a direct result (Scenario A)
        if callable(actual_result) and inner_data_paths:
            print("Detected closure pattern. Proceeding to inner function execution.")
            
            # Scenario B: evaluate_results returned an operator, test that operator
            for inner_path in inner_data_paths:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_result = inner_data.get('output')
                
                print(f"  Executing inner operator from {os.path.basename(inner_path)}...")
                actual_inner_result = actual_result(*inner_args, **inner_kwargs)
                
                # Verification
                passed, msg = recursive_check(expected_inner_result, actual_inner_result)
                if not passed:
                    print(f"FAILED: Inner function mismatch in {os.path.basename(inner_path)}")
                    print(msg)
                    sys.exit(1)
        else:
            # Scenario A: evaluate_results returned a value (or None), direct comparison
            # The provided evaluate_results implementation returns None (prints to stdout), 
            # so we check if the recorded output matches the actual output (likely None).
            
            passed, msg = recursive_check(expected_outer_result, actual_result)
            if not passed:
                print("FAILED: Output mismatch.")
                print(f"Expected: {expected_outer_result}")
                print(f"Actual: {actual_result}")
                print(msg)
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()