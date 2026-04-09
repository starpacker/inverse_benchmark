import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent__centered import _centered
from verification_utils import recursive_check

def test_centered():
    """
    Unit test for _centered function using captured data.
    """
    
    # 1. Define Data Paths
    # Based on the provided data_paths list, we only have the primary function data.
    # There are no secondary 'parent_function' files, indicating this is a direct execution (Scenario A).
    primary_data_path = '/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data__centered.pkl'

    # 2. Load Data
    if not os.path.exists(primary_data_path):
        print(f"Test Skipped: Data file not found at {primary_data_path}")
        sys.exit(0)

    print(f"Loading data from {primary_data_path}...")
    try:
        with open(primary_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load data using dill. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs and Expected Output
    # The dictionary structure is guaranteed by the data capture decorator
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output')

    print(f"Executing _centered with {len(args)} positional args and {len(kwargs)} keyword args.")

    # 4. Execute the Target Function
    try:
        actual_result = _centered(*args, **kwargs)
    except Exception as e:
        print("CRITICAL: Execution of _centered failed.")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify Results
    print("Verifying results against expected output...")
    try:
        # recursive_check handles numpy arrays, tensors, lists, dicts, and primitives
        passed, msg = recursive_check(expected_result, actual_result)
    except Exception as e:
        print(f"CRITICAL: Verification utility crashed. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED. Mismatch details: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_centered()