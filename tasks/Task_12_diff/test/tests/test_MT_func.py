import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_MT_func import MT_func
from verification_utils import recursive_check

# Hardcoded data paths as per instructions
data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_MT_func.pkl']

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Identify File Structure
    # Based on the provided path, we check if this is a factory pattern or a direct execution.
    # Looking at the function source:
    # def MT_func(x, H_fft): ... returns np.real(...)
    # It returns a value (array), not a callable. So this is Scenario A (Direct Execution).
    
    outer_path = None
    for p in data_paths:
        if 'standard_data_MT_func.pkl' in p:
            outer_path = p
            break

    if not outer_path:
        print("Error: standard_data_MT_func.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}")
    try:
        data = load_pkl(outer_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    # 2. Execute Function
    print("Executing MT_func...")
    try:
        actual_output = MT_func(*args, **kwargs)
    except Exception as e:
        print(f"Error executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
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