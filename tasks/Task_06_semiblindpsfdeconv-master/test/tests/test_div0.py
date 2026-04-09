import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

from agent_div0 import div0
from verification_utils import recursive_check

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_div0.pkl']
    
    # 2. Identify Data Files
    # For div0, which is a simple function returning a value (not a closure), 
    # we expect only the primary data file.
    primary_data_path = None
    
    for path in data_paths:
        if 'standard_data_div0.pkl' in path:
            primary_data_path = path
            break
            
    if not primary_data_path:
        print("Error: standard_data_div0.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from: {primary_data_path}")

    try:
        with open(primary_data_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution
    try:
        args = data_payload.get('args', [])
        kwargs = data_payload.get('kwargs', {})
        expected_result = data_payload.get('output')

        print("Executing div0...")
        actual_result = div0(*args, **kwargs)

    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        
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
    run_test()