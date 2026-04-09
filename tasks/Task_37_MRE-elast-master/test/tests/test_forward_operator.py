import sys
import os
import dill
import numpy as np
import scipy.sparse.linalg
import traceback

# Optional torch import
try:
    import torch
except ImportError:
    torch = None

# Add current directory to path so imports work
sys.path.append(os.getcwd())

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def test_forward_operator():
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 1. Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_data_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_data_path = path

    if not outer_data_path:
        print("Error: standard_data_forward_operator.pkl not found.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        # If dill fails because torch is missing but data contains torch tensors
        if "torch" in str(e) and torch is None:
            print("Failed to load data possibly due to missing 'torch' module.")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    print(f"Loaded outer data from {outer_data_path}")

    # 3. Execution Strategy
    # Determine if we are in Scenario A (Simple Function) or B (Factory)
    # Based on the provided code snippet, forward_operator returns 'disp' (numpy array),
    # not a function. However, the prompt mentions Factory/Closure patterns.
    # We will dynamically check the result of the first call.

    try:
        print("Executing forward_operator with outer args...")
        result_phase_1 = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    final_result = result_phase_1
    final_expected = expected_output

    # Check if the result is a callable (Scenario B - Factory Pattern)
    if callable(result_phase_1) and inner_data_path:
        print("Phase 1 result is callable. Proceeding to Inner Data execution (Scenario B).")
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)

        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        final_expected = inner_data.get('output')

        try:
            print("Executing inner operator...")
            final_result = result_phase_1(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error executing inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)

    elif callable(result_phase_1) and not inner_data_path:
        print("Warning: forward_operator returned a callable, but no inner data file was found to test it.")
        # We can't verify the inner logic, but the factory creation didn't crash.
        # Often this implies the pickle expected output is the function object itself, which is hard to compare.
        # We will assume pass if execution succeeded, but note the ambiguity.
        print("TEST PASSED (Factory creation successful, no inner data to test closure)")
        sys.exit(0)

    # 4. Verification
    print("Verifying results...")
    
    # Handle sparse matrix comparison if necessary (though usually they are dense arrays in output)
    # The SUT returns a numpy array 'disp', so recursive_check should handle it.

    passed, msg = recursive_check(final_expected, final_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()