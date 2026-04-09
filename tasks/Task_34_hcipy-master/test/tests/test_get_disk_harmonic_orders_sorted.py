import sys
import os
import dill
import traceback
import numpy as np

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_get_disk_harmonic_orders_sorted import get_disk_harmonic_orders_sorted
from verification_utils import recursive_check

def test_get_disk_harmonic_orders_sorted():
    """
    Unit test for get_disk_harmonic_orders_sorted using recorded standard data.
    """
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_get_disk_harmonic_orders_sorted.pkl']
    
    # Identify the relevant data file
    # In this case, since get_disk_harmonic_orders_sorted returns a list (not a function),
    # we expect a single file containing the direct input/output mapping.
    outer_path = None
    for path in data_paths:
        if 'standard_data_get_disk_harmonic_orders_sorted.pkl' in path:
            outer_path = path
            break
            
    if not outer_path or not os.path.exists(outer_path):
        print(f"Test Skipped: Data file not found at {outer_path}")
        sys.exit(0)

    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract arguments and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    print(f"Running test for function: {data.get('func_name')}")
    print(f"Input Args keys: {len(args)}")
    print(f"Input Kwargs keys: {list(kwargs.keys())}")

    try:
        # Execute the function
        actual_result = get_disk_harmonic_orders_sorted(*args, **kwargs)
        
        # Verify the result
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_get_disk_harmonic_orders_sorted()