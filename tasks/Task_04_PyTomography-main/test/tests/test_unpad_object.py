import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent code to the path if necessary
# sys.path.append('/path/to/codebase') 

from agent_unpad_object import unpad_object
from verification_utils import recursive_check

# Data paths provided in the prompt
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_unpad_object.pkl']

def run_test():
    # 1. Identify Data File
    # Based on the function name 'unpad_object', we look for the main data file.
    # Since the input list only contains one file and the function returns a Tensor (not a callable),
    # this falls under Scenario A (Simple Function).
    target_path = None
    for p in data_paths:
        if 'standard_data_unpad_object.pkl' in p:
            target_path = p
            break
    
    if not target_path:
        print("Error: standard_data_unpad_object.pkl not found in data_paths.")
        sys.exit(1)

    print(f"Loading data from {target_path}...")
    
    # 2. Load Data
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading dill file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Arguments and Expected Output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    print("Executing unpad_object...")
    
    # 4. Execute Function
    try:
        actual_result = unpad_object(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    # The function unpad_object returns a torch.Tensor (sliced), so we compare directly against expected output.
    print("Verifying result...")
    
    passed, msg = recursive_check(expected_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()