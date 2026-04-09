import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the current directory is in the path to import local modules
sys.path.append(os.getcwd())

from agent_compute_grid_in_memory import compute_grid_in_memory
from verification_utils import recursive_check

# Data paths provided for the test
data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_compute_grid_in_memory.pkl']

def main():
    print("Starting test for compute_grid_in_memory...")

    # 1. File Logic Setup: Identify the standard data file
    # Since the function returns a list (data), not a closure, we expect Scenario A.
    file_path = None
    for path in data_paths:
        if 'standard_data_compute_grid_in_memory.pkl' in path:
            file_path = path
            break
            
    if file_path is None or not os.path.exists(file_path):
        print(f"Error: Data file not found in paths: {data_paths}")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Extract Args/Kwargs and Expected Output
    input_args = data_payload.get('args', [])
    input_kwargs = data_payload.get('kwargs', {})
    expected_result = data_payload.get('output')

    # 3. Execution
    print("Executing compute_grid_in_memory with loaded inputs...")
    try:
        # Scenario A: Simple Function Call
        actual_result = compute_grid_in_memory(*input_args, **input_kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Comparing actual result with expected result...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
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
    main()