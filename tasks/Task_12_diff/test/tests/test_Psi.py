import sys
import os
import dill
import numpy as np
import traceback

# Optional import for torch to handle potential data types, but not strictly required for Psi logic
try:
    import torch
except ImportError:
    torch = None

from agent_Psi import Psi
from verification_utils import recursive_check

def test_psi():
    """
    Test script for the function Psi using serialized data.
    """
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_Psi.pkl']
    
    # 1. Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'standard_data_Psi.pkl' in path:
            outer_data_path = path
        elif 'parent_function_Psi' in path:
            inner_data_path = path

    if not outer_data_path:
        print("Error: standard_data_Psi.pkl not found in data_paths.")
        sys.exit(1)

    # 2. Load Outer Data (Arguments to create the operator or result directly)
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_data_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. Execute Target Function
    try:
        # Psi is a direct function returning a numpy array, not a factory pattern based on the provided code.
        # However, we check if the result is callable just in case the context implies a factory pattern (Scenario B).
        actual_result = Psi(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing Psi: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine Validation Strategy
    # Scenario B: Psi returns a callable (factory pattern) and we have inner data to test that callable.
    if callable(actual_result) and inner_data_path:
        print("Detected Factory Pattern (Scenario B). Loading inner data...")
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)

        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_inner_output = inner_data.get('output', None)

        try:
            final_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error executing inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected_to_check = expected_inner_output
        actual_to_check = final_result

    # Scenario A: Psi returns the final result directly.
    else:
        print("Detected Direct Execution (Scenario A).")
        expected_to_check = expected_outer_output
        actual_to_check = actual_result

    # 5. Verification
    try:
        passed, msg = recursive_check(expected_to_check, actual_to_check)
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
    test_psi()