import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_conditioning import conditioning
from verification_utils import recursive_check

# Helper function to load data safely
def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_conditioning.pkl']
    
    outer_path = None
    inner_paths = []

    # Classify files based on naming convention
    for p in data_paths:
        if 'standard_data_conditioning.pkl' in p:
            outer_path = p
        elif 'parent_function_conditioning' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_conditioning.pkl not found in data paths.")
        sys.exit(1)

    try:
        # 2. Load Outer Data (Main Function Inputs)
        print(f"Loading outer data from {outer_path}")
        outer_data = load_data(outer_path)
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        # 3. Execute Target Function
        print("Executing conditioning with outer args/kwargs...")
        actual_result = conditioning(*outer_args, **outer_kwargs)

        # 4. Handle Scenarios (Closure vs Simple Function)
        # Scenario B: Closure Pattern (Inner files exist)
        if inner_paths:
            print("Scenario B: Detected Closure/Factory pattern. Validating inner execution...")
            
            if not callable(actual_result):
                print(f"Error: Expected 'conditioning' to return a callable (operator), but got {type(actual_result)}.")
                sys.exit(1)

            operator = actual_result
            
            # Test each inner capture file
            for inner_path in inner_paths:
                print(f"  Testing inner capture: {inner_path}")
                inner_data = load_data(inner_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)

                # Execute the closure/operator
                inner_actual = operator(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(inner_expected, inner_actual)
                if not passed:
                    print(f"  FAILED validation for {inner_path}")
                    print(f"  Diff: {msg}")
                    sys.exit(1)
                else:
                    print("  Inner validation passed.")

        # Scenario A: Simple Function (No inner files)
        else:
            print("Scenario A: Simple function execution. Validating direct output...")
            
            # Note: recursive_check handles complex types (lists, tuples, numpy arrays) automatically
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print("FAILED validation for direct output.")
                print(f"Diff: {msg}")
                sys.exit(1)
            else:
                print("Direct output validation passed.")

        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print("An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()