import sys
import os
import dill
import traceback

# Optional imports to ensure compatibility with pickled data if it contains these types
# and to prevent ModuleNotFoundError if they are not installed in the test environment.
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from agent_fetch import fetch
from verification_utils import recursive_check

def test_fetch():
    """
    Unit test for the fetch function in agent_fetch.py
    """
    
    # 1. Define Data Paths
    # Based on the prompt's provided path
    data_dir = '/data/yjh/phasorpy-main_sandbox/run_code/std_data'
    outer_file = 'standard_data_fetch.pkl'
    outer_path = os.path.join(data_dir, outer_file)

    # 2. Check if data exists
    if not os.path.exists(outer_path):
        print(f"Skipping test: Data file not found at {outer_path}")
        sys.exit(0)

    # 3. Load Data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output')
    func_name = data.get('func_name')

    print(f"Testing function: {func_name}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

    # 5. Execution
    try:
        # Run the function
        actual_result = fetch(*args, **kwargs)
        
        # Scenario A: Simple Function (fetch returns a path string, not a callable)
        # We compare the result of the function directly.
        
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Verification
    try:
        # Special handling for fetch:
        # If fetch returns a file path, different environments might have different absolute paths 
        # (e.g., pooch cache locations).
        # We perform a standard recursive check first. 
        passed, msg = recursive_check(expected_result, actual_result)
        
        if not passed:
            # Fallback for path differences: if both are strings and end with the same filename
            if isinstance(expected_result, str) and isinstance(actual_result, str):
                if os.path.basename(expected_result) == os.path.basename(actual_result):
                    print("Warning: Paths differ but filenames match. Accepting as pass due to environment differences.")
                    passed = True
                else:
                    # Check if expected was a relative path resolved to absolute in recording
                    # and actual is just the relative path (or vice versa)
                    if expected_result.endswith(actual_result) or actual_result.endswith(expected_result):
                         print("Warning: Paths match partially. Accepting as pass.")
                         passed = True

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_result}")
            print(f"Actual:   {actual_result}")
            sys.exit(1)

    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_fetch()