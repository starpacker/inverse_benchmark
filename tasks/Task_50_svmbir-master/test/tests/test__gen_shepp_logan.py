import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the verification utility is accessible
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: 'verification_utils.py' not found. Please ensure it is in the path.")
    sys.exit(1)

# Import the target function
try:
    from agent__gen_shepp_logan import _gen_shepp_logan
except ImportError:
    print("Error: 'agent__gen_shepp_logan.py' not found or function missing.")
    sys.exit(1)

def main():
    # 1. Configuration
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data__gen_shepp_logan.pkl']
    
    # 2. Identify the Outer Data File (Standard Function Call)
    # Since we only have one path and it matches the function name directly, this is a standard execution.
    outer_path = None
    for path in data_paths:
        if 'standard_data__gen_shepp_logan.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("Error: Standard data file 'standard_data__gen_shepp_logan.pkl' not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs and Expected Outputs
    try:
        func_name = outer_data.get('func_name', '')
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Target Function: {func_name}")
        # print(f"Args: {args}")
        # print(f"Kwargs: {kwargs}")

    except Exception as e:
        print(f"Error extracting data from payload: {e}")
        sys.exit(1)

    # 4. Execute the Function
    print("Running _gen_shepp_logan...")
    try:
        actual_result = _gen_shepp_logan(*args, **kwargs)
    except Exception as e:
        print(f"Error executing _gen_shepp_logan: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()