import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Ensure the target module is in the path
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_paths = []

    # 2. Categorize Data Files
    for path in data_paths:
        if 'parent_function' in path:
            inner_paths.append(path)
        elif path.endswith('standard_data_evaluate_results.pkl'):
            outer_path = path

    if not outer_path:
        print("Error: No outer data file 'standard_data_evaluate_results.pkl' found.")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_path}")
    
    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file {outer_path}: {e}")
        sys.exit(1)

    # 4. Execute Target Function (Outer Layer)
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Suppress plot output for testing
        plt.switch_backend('Agg') 
        
        print("Executing 'evaluate_results' with loaded arguments...")
        actual_result = evaluate_results(*args, **kwargs)
        
    except Exception:
        traceback.print_exc()
        print("Error: Execution of 'evaluate_results' failed.")
        sys.exit(1)

    # 5. Determine Verification Strategy (Simple vs Factory)
    
    # Scenario B: Factory Pattern (If inner files exist)
    if inner_paths:
        print(f"Detected Factory Pattern. {len(inner_paths)} inner data files found.")
        
        if not callable(actual_result):
            print(f"Error: Expected 'evaluate_results' to return a callable (factory pattern), but got {type(actual_result)}.")
            sys.exit(1)
            
        operator = actual_result
        
        for inner_path in inner_paths:
            print(f"Testing Inner Data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                actual_inner_result = operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not passed:
                    print(f"FAILED: Validation mismatch for inner execution {inner_path}.")
                    print(msg)
                    sys.exit(1)
                    
            except Exception:
                traceback.print_exc()
                print(f"Error: Execution of inner closure failed for {inner_path}.")
                sys.exit(1)

    # Scenario A: Simple Function (Direct Result)
    else:
        print("Detected Simple Function Pattern.")
        expected_output = outer_data.get('output')
        
        # Verify Results
        passed, msg = recursive_check(expected_output, actual_result)
        
        if not passed:
            print("FAILED: Result mismatch.")
            print(msg)
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()