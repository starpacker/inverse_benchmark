import sys
import os
import dill
import numpy as np
import traceback
import torch

# Add the directory containing the function to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function to be tested
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def test_forward_operator():
    """
    Test script for forward_operator.
    Strategy:
    1. Load the data pickle.
    2. Determine if it's a direct function call or a factory pattern based on the data.
       (Based on the provided path, it seems to be a direct function call, but we handle both).
    3. Execute the function.
    4. Compare results.
    """
    
    # Path provided in the prompt
    data_path = '/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)
    
    print(f"Function inputs loaded. Args length: {len(args)}, Kwargs keys: {list(kwargs.keys())}")

    # Execution
    print("Executing forward_operator...")
    try:
        # Based on the function definition in context, forward_operator returns a result (ksp_masked),
        # not a callable. It is a direct computation function.
        actual_output = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verification
    print("Verifying results...")
    
    # Check if the output is valid compared to expected
    try:
        is_match, failure_msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"Error during recursive check: {e}")
        traceback.print_exc()
        sys.exit(1)

    if is_match:
        print("TEST PASSED: Output matches expected data.")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {failure_msg}")
        # Detailed debugging info
        if isinstance(expected_output, np.ndarray):
            print(f"Expected shape: {expected_output.shape}, Actual shape: {actual_output.shape}")
            print(f"Expected dtype: {expected_output.dtype}, Actual dtype: {actual_output.dtype}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()