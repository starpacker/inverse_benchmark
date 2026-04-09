import sys
import os
import dill
import numpy as np
import torch
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_forward_operator import forward_operator
except ImportError:
    print("Error: Could not import 'forward_operator' from 'agent_forward_operator.py'")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'")
    sys.exit(1)

def test_forward_operator():
    # 1. DATA FILE ANALYSIS
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_path = path

    if not outer_path:
        print("Error: Standard data file for 'forward_operator' not found.")
        sys.exit(1)

    # 2. LOAD DATA
    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. EXECUTION & VERIFICATION
    # The function forward_operator returns a list of arrays (Scenario A - Simple Function)
    # It does not appear to be a closure/factory based on the reference code provided.
    
    print(f"Executing forward_operator with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
    
    try:
        actual_output = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error during execution of forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. If an inner path existed (Scenario B), we would use actual_output as a callable here.
    # However, based on the reference code, forward_operator returns [I0, I1, I2, I3], which is data.
    # Therefore, we compare immediately.

    print("Verifying results...")
    
    # Handle potential differences in container types (list vs tuple) implicitly handled by recursive_check
    # or ensure they match if strict.
    
    passed, msg = recursive_check(expected_output, actual_output)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()