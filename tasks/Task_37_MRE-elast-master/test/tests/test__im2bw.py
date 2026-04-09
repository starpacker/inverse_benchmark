import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch dependency
try:
    import torch
except ImportError:
    torch = None

# Add current directory to path so we can import the target function
sys.path.append(os.path.dirname(__file__))

# Import the target function
from agent__im2bw import _im2bw
from verification_utils import recursive_check

def test_im2bw():
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data__im2bw.pkl']
    
    # 1. Identify File Roles
    # Since _im2bw is a simple function (not a factory), we expect a direct execution.
    # We look for the main data file.
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        else:
            outer_path = p

    if not outer_path:
        print("Error: No standard_data__im2bw.pkl found.")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        sys.exit(1)

    # 3. Execution Strategy
    # Scenario A: Simple Function (Direct Execution)
    # The function _im2bw(Ig, level) returns a result directly. 
    # It does not return another function.
    
    try:
        print(f"Running _im2bw with args from {os.path.basename(outer_path)}...")
        
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')

        # Execute target function
        actual_output = _im2bw(*args, **kwargs)

        # 4. Verification
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_im2bw()