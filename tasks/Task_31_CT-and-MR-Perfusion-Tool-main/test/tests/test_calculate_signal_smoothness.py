import sys
import os
import dill
import numpy as np
import traceback

# Handle conditional torch import
try:
    import torch
except ImportError:
    torch = None

from agent_calculate_signal_smoothness import calculate_signal_smoothness
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    # The prompt provided specific paths. We only have the standard data for the function itself,
    # indicating a simple function call, not a factory pattern.
    data_path = '/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_calculate_signal_smoothness.pkl'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs and Expected Output
    # The data structure from the decorator is:
    # {'func_name': ..., 'args': ..., 'kwargs': ..., 'output': ...}
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output')
        
        # Debugging info
        print(f"Function: {data.get('func_name')}")
        # print(f"Args: {args}") # Uncomment if args are small enough to print
        # print(f"Kwargs: {kwargs}")
    except Exception as e:
        print(f"Error extracting data structure: {e}")
        sys.exit(1)

    # 4. Execute the Function
    try:
        print("Executing calculate_signal_smoothness...")
        actual_result = calculate_signal_smoothness(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    try:
        # Using recursive_check from verification_utils as requested
        is_correct, msg = recursive_check(expected_output, actual_result)
        
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_output}")
            print(f"Actual:   {actual_result}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()