import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def run_test():
    """
    Test script for load_and_preprocess_data.
    Handles both direct execution (Scenario A) and factory/closure patterns (Scenario B).
    """
    
    # 1. Identify Outer and Inner Data
    outer_path = None
    inner_paths = []
    
    # Standard naming convention for the primary function data
    target_func_name = "load_and_preprocess_data"
    expected_outer_suffix = f"standard_data_{target_func_name}.pkl"
    
    for p in data_paths:
        if p.endswith(expected_outer_suffix):
            outer_path = p
        elif "parent_function" in p and target_func_name in p:
            inner_paths.append(p)
            
    if not outer_path:
        print(f"Skipping test: No outer data file found for {target_func_name} in provided paths.")
        # If no data exists, we cannot test. This might be valid in some pipelines, but usually implies a setup error.
        # We exit 0 to avoid breaking CI if data generation was skipped intentionally, 
        # but printing a warning is crucial.
        sys.exit(0)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get("args", [])
    outer_kwargs = outer_data.get("kwargs", {})
    expected_outer_output = outer_data.get("output", None)

    # 2. Run the Primary Function (The "Outer" Execution)
    print(f"Executing {target_func_name} with retrieved args/kwargs...")
    try:
        # Check if the environment requires specific handling for missing torch if data contained tensors
        # (Though we removed torch import, dill might try to unpickle torch objects if they exist in data)
        actual_outer_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution of {target_func_name} failed:")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Strategy based on Output Type (Callable vs Data)
    if callable(actual_outer_result) and inner_paths:
        print("Detected Factory Pattern (Scenario B). Testing inner execution(s)...")
        # Scenario B: The function returned a callable (operator), and we have inner data files.
        operator = actual_outer_result
        
        for i_path in inner_paths:
            print(f"Testing inner data: {i_path}")
            try:
                with open(i_path, "rb") as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Failed to load inner data {i_path}: {e}")
                continue # Try next inner file

            inner_args = inner_data.get("args", [])
            inner_kwargs = inner_data.get("kwargs", {})
            expected_inner_output = inner_data.get("output")

            try:
                actual_inner_result = operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner execution failed for {i_path}:")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify Inner Result
            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"Verification FAILED for inner data {i_path}")
                print(msg)
                sys.exit(1)
            else:
                print(f"Inner verification PASSED for {i_path}")

    else:
        print("Detected Simple Execution (Scenario A). Verifying direct output...")
        # Scenario A: The function returned data directly, or it's a factory but no inner data was recorded.
        # We compare the result of the outer execution against the recorded outer output.
        
        passed, msg = recursive_check(expected_outer_output, actual_outer_result)
        if not passed:
            print("Verification FAILED for outer execution.")
            print(msg)
            # Detailed debug for Tiff handling differences if common
            if isinstance(expected_outer_output, tuple) and isinstance(actual_outer_result, tuple):
                print(f"Expected tuple len: {len(expected_outer_output)}, Actual: {len(actual_outer_result)}")
            sys.exit(1)
        
        print("Verification PASSED.")

    sys.exit(0)

if __name__ == "__main__":
    run_test()