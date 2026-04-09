import sys
import os
import dill
import torch
import numpy as np
import traceback
import warnings

# Add current directory to path so imports work
sys.path.append(os.path.dirname(__file__))

# Import the target function
from agent_resample_kernel import resample_kernel
from verification_utils import recursive_check

def test_resample_kernel():
    # 1. Configuration and Data Paths
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_resample_kernel.pkl']
    
    # Analyze files to distinguish between Scenario A (Simple) and B (Factory/Closure)
    outer_path = None
    inner_paths = []

    for path in data_paths:
        filename = os.path.basename(path)
        if "parent_function" in filename:
            inner_paths.append(path)
        else:
            outer_path = path

    print(f"Test Configuration:")
    print(f"  Outer Data Path: {outer_path}")
    print(f"  Inner Data Paths: {len(inner_paths)}")

    if not outer_path or not os.path.exists(outer_path):
        print("Error: Primary data file (outer) not found.")
        sys.exit(1)

    try:
        # 2. Phase 1: Run the Outer Function
        print("\n--- Phase 1: Running Primary Function ---")
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_result = outer_data.get('output')

        print(f"  Loaded args type: {[type(x) for x in outer_args]}")
        print(f"  Loaded kwargs keys: {outer_kwargs.keys()}")

        # Execute the function
        actual_result = resample_kernel(*outer_args, **outer_kwargs)
        
        # 3. Phase 2: Verification Logic
        # Determine if we need to check the result directly (Scenario A) or execute a returned closure (Scenario B)
        
        if len(inner_paths) > 0:
            print("\n--- Phase 2: Testing Returned Operator (Factory Pattern) ---")
            # Scenario B: The outer function returned a callable, test it against inner data
            if not callable(actual_result):
                print(f"Error: Inner data files exist, implying a factory pattern, but the result is not callable. Type: {type(actual_result)}")
                sys.exit(1)
            
            for inner_path in inner_paths:
                print(f"  Testing inner data: {os.path.basename(inner_path)}")
                with open(inner_path, "rb") as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_result = inner_data.get('output')

                # Execute the closure/operator
                closure_result = actual_result(*inner_args, **inner_kwargs)

                # Verify
                passed, msg = recursive_check(expected_inner_result, closure_result)
                if not passed:
                    print(f"FAILED: Closure execution mismatch in file {os.path.basename(inner_path)}")
                    print(msg)
                    sys.exit(1)
            
            print("  All inner factory tests passed.")

        else:
            print("\n--- Phase 2: Verifying Direct Result ---")
            # Scenario A: The outer function returned the final data
            passed, msg = recursive_check(expected_outer_result, actual_result)
            if not passed:
                print("FAILED: Direct result mismatch.")
                print(msg)
                sys.exit(1)
            print("  Direct result verification passed.")

    except Exception as e:
        print(f"\nCRITICAL ERROR during execution:")
        traceback.print_exc()
        sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_resample_kernel()