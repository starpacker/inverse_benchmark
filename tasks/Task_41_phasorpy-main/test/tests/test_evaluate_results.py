import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function to test
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def load_dill(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    print("Starting test_evaluate_results.py...")
    
    # 1. Identify Outer and Inner Data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_evaluate_results.pkl' in p:
            outer_path = p
        elif 'parent_function_evaluate_results' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_evaluate_results.pkl not found in data_paths.")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_path}")
    try:
        outer_data = load_dill(outer_path)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Execute Outer Function
    print("Executing evaluate_results with outer arguments...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the agent function
        result_or_operator = evaluate_results(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Scenario and Verify
    if inner_paths:
        # Scenario B: Factory Pattern (Function returned a callable)
        print(f"Detected {len(inner_paths)} inner data files. Testing Factory/Closure pattern.")
        
        if not callable(result_or_operator):
            print(f"Error: Inner data exists, implying Factory pattern, but evaluate_results returned {type(result_or_operator)} instead of callable.")
            sys.exit(1)

        for i_path in inner_paths:
            print(f"  Testing inner file: {i_path}")
            try:
                inner_data = load_dill(i_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')

                # Execute the closure/operator
                actual_inner_result = result_or_operator(*inner_args, **inner_kwargs)

                # Verify
                is_correct, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not is_correct:
                    print(f"  FAILED validation for inner file {i_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  PASSED validation for inner file {i_path}")

            except Exception as e:
                print(f"  Error processing inner file {i_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple Function (Direct result comparison)
        print("No inner data files found. Testing Simple Function pattern.")
        expected_output = outer_data.get('output')
        
        is_correct, msg = recursive_check(expected_output, result_or_operator)
        if not is_correct:
            print("FAILED validation for evaluate_results (Simple Pattern)")
            print(f"Message: {msg}")
            sys.exit(1)
        else:
            print("PASSED validation for evaluate_results (Simple Pattern)")

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()