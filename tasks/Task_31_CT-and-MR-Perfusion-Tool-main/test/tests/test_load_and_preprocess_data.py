import sys
import os
import dill
import numpy as np
import traceback
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
# Import verification utility
from verification_utils import recursive_check

# Define data paths provided in instructions
data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def main():
    print("----------------------------------------------------------------")
    print("Test Script: test_load_and_preprocess_data.py")
    print("Target Function: load_and_preprocess_data")
    print("----------------------------------------------------------------")

    # 1. Identify Data Files
    # We look for the primary data file corresponding to the function execution
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_load_and_preprocess_data_' in p:
            inner_paths.append(p)

    if not outer_path:
        print("ERROR: Standard data file 'standard_data_load_and_preprocess_data.pkl' not found.")
        sys.exit(1)

    print(f"Loading test data from: {outer_path}")

    # 2. Load Data and Execute
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing load_and_preprocess_data with loaded arguments...")
        
        # Scenario A: Simple Function Execution
        # Based on the provided code and data paths, this is likely a direct function call 
        # rather than a factory pattern returning a closure, as there are no inner data files.
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)

        # 3. Verification
        print("Verifying results...")
        
        # Check if the result is a closure/function (Scenario B check)
        # If the function was a factory, actual_result would be callable, and we'd need inner files.
        # Given the provided paths, we assume Scenario A (direct result).
        
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("Verification Successful: Actual result matches expected output.")
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"Verification Failed: {msg}")
            print("TEST FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()