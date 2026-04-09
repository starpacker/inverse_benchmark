import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent__find_origin_by_convolution import _find_origin_by_convolution
from verification_utils import recursive_check

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data__find_origin_by_convolution.pkl']
    
    target_file = None
    for path in data_paths:
        if 'standard_data__find_origin_by_convolution.pkl' in path:
            target_file = path
            break
            
    if not target_file or not os.path.exists(target_file):
        print(f"Test Skipped: Data file not found in {data_paths}")
        sys.exit(0) # Exit 0 as it's not a logic failure, just missing data context

    # 2. Load Data
    try:
        with open(target_file, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Running test for: _find_origin_by_convolution")
    print(f"Input file: {target_file}")

    # 3. Execution
    try:
        # The function _find_origin_by_convolution calculates the origin and returns a tuple/list directly.
        # It is not a factory function, so we execute it once.
        actual_result = _find_origin_by_convolution(*args, **kwargs)
        
    except Exception as e:
        print(f"Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_output}")
            print(f"Actual:   {actual_result}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()