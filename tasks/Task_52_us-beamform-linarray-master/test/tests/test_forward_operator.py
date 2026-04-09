import sys
import os
import dill
import numpy as np
import traceback
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def run_test():
    # 1. Identify Data Files
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function' in path:
            inner_path = path
    
    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execute Function
    try:
        # Based on the provided code, forward_operator is a standard function, not a factory/decorator itself in the core logic 
        # (though it might have been wrapped by decorators during generation).
        # We need to determine if we are in Scenario A (standard function) or B (closure factory).
        
        # Checking if there is an inner path implies a factory pattern in the data generation logic.
        # If inner_path is None, we treat it as a direct function call (Scenario A).
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_result = outer_data.get('output', None)

        print("Executing forward_operator with loaded arguments...")
        result = forward_operator(*outer_args, **outer_kwargs)
        
        # Scenario B Handling: If the result is callable and we have inner data, invoke it.
        if callable(result) and inner_path:
            print(f"Result is callable (Factory pattern detected). Loading inner data from {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_result = inner_data.get('output', None)
            
            # Execute inner function
            print("Executing inner operator...")
            actual_result = result(*inner_args, **inner_kwargs)
            expected_result = expected_inner_result
            
        else:
            # Scenario A: Result is the final data
            actual_result = result
            expected_result = expected_outer_result

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()