import sys
import os
import dill
import numpy as np
import traceback

# Safe import for torch, as it might not be available in all environments
try:
    import torch
except ImportError:
    torch = None

from agent__ggradient import _ggradient
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data__ggradient.pkl']

    # 2. Analyze Data Files
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        elif 'standard_data__ggradient.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: Standard data file for _ggradient not found.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 4. Phase 1: Execute Target Function
    try:
        print("Executing _ggradient with outer arguments...")
        actual_result_or_operator = _ggradient(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing _ggradient: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Determine Scenario (Factory vs Simple Function)
    is_operator = callable(actual_result_or_operator)
    
    final_result = None
    expected_result = None

    if is_operator and inner_path:
        # Scenario B: Factory Pattern with inner data available
        print("Function returned a callable and inner data found. Proceeding with Factory execution.")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output', None)

            final_result = actual_result_or_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error executing inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    elif is_operator and not inner_path:
        # Edge case: Returns callable but no data to test it. 
        # We assume the test is just to ensure it returns the callable as expected?
        # Usually checking identity or basic property. For now, we compare against the outer output (which is the callable itself usually)
        print("Function returned a callable but no inner data found. Comparing operator identity/structure.")
        final_result = actual_result_or_operator
        expected_result = expected_outer_output

    else:
        # Scenario A: Simple Function (Result is value)
        print("Function returned a value. Proceeding with value comparison.")
        final_result = actual_result_or_operator
        expected_result = expected_outer_output

    # 6. Verification
    try:
        passed, msg = recursive_check(expected_result, final_result)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()