import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function and verification utility
from agent_PS4 import PS4
from verification_utils import recursive_check

def run_test():
    # Define data paths
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_PS4.pkl']
    
    # Identify Outer and Inner data files based on the file naming convention
    # PS4 is likely a standard function, not a factory, based on the signature provided,
    # but we will handle both Scenario A and Scenario B just in case.
    
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'standard_data_PS4.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_PS4_' in path:
            inner_paths.append(path)
            
    if not outer_path:
        print("Error: standard_data_PS4.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # --- Phase 1: Execute Outer Function ---
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    print("Executing PS4 with outer arguments...")
    try:
        actual_outer_result = PS4(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Error executing PS4:")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine if Factory Pattern or Simple Function ---
    
    # Check if there are inner paths (Factory Pattern / Closure)
    if inner_paths:
        print(f"Detected {len(inner_paths)} inner data files. Treating PS4 as a factory function.")
        
        # Verify the result is callable
        if not callable(actual_outer_result):
            print("Error: PS4 was expected to return a callable (operator) based on the presence of inner data files, but it did not.")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"  Testing Inner Data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"  Error loading inner pickle file: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)

            try:
                actual_inner_result = actual_outer_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print("  Error executing inner operator:")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare Inner Result
            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"  FAILED: {msg}")
                sys.exit(1)
            else:
                print("  Inner Verification: OK")

    else:
        # Scenario A: Simple Function (Likely the case for PS4 based on signature)
        print("No inner data files detected. Treating PS4 as a standard function.")
        
        passed, msg = recursive_check(expected_outer_output, actual_outer_result)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("Verification: OK")

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()