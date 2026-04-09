import sys
import os
import dill
import numpy as np
import traceback

# Conditional import for torch to handle environments where it's missing
try:
    import torch
except ImportError:
    torch = None

from agent_SoftThresh import SoftThresh
from verification_utils import recursive_check

def run_test():
    # Paths provided in the prompt context
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_SoftThresh.pkl']
    
    # Identify the relevant data file
    target_file = None
    for path in data_paths:
        if 'standard_data_SoftThresh.pkl' in path:
            target_file = path
            break
            
    if not target_file:
        print("Skipping test: standard_data_SoftThresh.pkl not found in paths.")
        # If no data is present, we can't test, but we shouldn't fail the pipeline if it's just missing data.
        # However, for this specific request, we expect data.
        sys.exit(0)

    try:
        # Load the data
        if not os.path.exists(target_file):
            print(f"Error: File {target_file} does not exist.")
            sys.exit(1)
            
        with open(target_file, 'rb') as f:
            data = dill.load(f)
            
        # Extract inputs and expected output
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output')
        
        # Execute the function
        # Based on the signature def SoftThresh(x, tau): ... and the fact it returns an array,
        # this is a direct execution, not a factory pattern.
        actual_result = SoftThresh(*args, **kwargs)
        
        # Verify results
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        traceback.print_exc()
        print(f"TEST FAILED with Exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()