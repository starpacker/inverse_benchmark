import sys
import os
import dill
import numpy as np
import scipy.fft
import traceback
from agent_filter_sinogram import filter_sinogram
from verification_utils import recursive_check

# List of all provided data paths
data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_filter_sinogram.pkl']

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # Identify Scenario: Simple Function vs Factory Pattern
    # 1. Look for the main entry point data
    outer_path = None
    for p in data_paths:
        if p.endswith("standard_data_filter_sinogram.pkl"):
            outer_path = p
            break
            
    if not outer_path:
        print("TEST FAILED: Standard data file 'standard_data_filter_sinogram.pkl' not found.")
        sys.exit(1)

    # 2. Look for inner/closure data files (indicating a factory pattern)
    inner_paths = [p for p in data_paths if "parent_function_filter_sinogram" in p]
    
    try:
        # --- SCENARIO A: Simple Function (One-shot execution) ---
        if not inner_paths:
            print(f"Loading data from {outer_path}...")
            data = load_data(outer_path)
            
            args = data.get('args', [])
            kwargs = data.get('kwargs', {})
            expected_output = data.get('output')
            
            print("Executing filter_sinogram...")
            actual_output = filter_sinogram(*args, **kwargs)
            
            print("Verifying result...")
            is_match, msg = recursive_check(expected_output, actual_output)
            
            if is_match:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: Output mismatch.\n{msg}")
                sys.exit(1)

        # --- SCENARIO B: Factory Pattern (Closure execution) ---
        else:
            # Step 1: Create the Operator/Closure using Outer Data
            print(f"Loading initialization data from {outer_path}...")
            outer_data = load_data(outer_path)
            outer_args = outer_data.get('args', [])
            outer_kwargs = outer_data.get('kwargs', {})
            
            print("Initializing operator via filter_sinogram...")
            # Ideally, filter_sinogram returns a function here
            operator = filter_sinogram(*outer_args, **outer_kwargs)
            
            if not callable(operator):
                print("TEST FAILED: Expected filter_sinogram to return a callable (factory pattern), but got value.")
                # Fallback check if it was actually Scenario A misidentified, though unlikely if inner files exist
                is_match, msg = recursive_check(outer_data['output'], operator)
                if is_match:
                    print("Wait, direct output matched outer data despite inner files existing. Ambiguous state.")
                sys.exit(1)

            # Step 2: Execute the Operator using Inner Data
            print(f"Found {len(inner_paths)} inner execution files. Testing all...")
            
            for inner_path in inner_paths:
                print(f"  Testing with {inner_path}...")
                inner_data = load_data(inner_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                # Run the closure
                actual_inner_output = operator(*inner_args, **inner_kwargs)
                
                is_match, msg = recursive_check(expected_inner_output, actual_inner_output)
                if not is_match:
                    print(f"TEST FAILED on inner file {os.path.basename(inner_path)}.\n{msg}")
                    sys.exit(1)

            print("TEST PASSED")
            sys.exit(0)

    except Exception as e:
        print(f"TEST FAILED: Exception during execution.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()