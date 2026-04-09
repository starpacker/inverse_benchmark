import sys
import os
import dill
import numpy as np
import traceback
import warnings

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_rotate_kernel import rotate_kernel
except ImportError:
    print("Error: Could not import 'rotate_kernel' from 'agent_rotate_kernel.py'. Make sure the file exists.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing, though instructions imply it exists.
    # We will define a minimal version if needed, but per instructions, we assume it exists.
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def run_test():
    # Data paths provided in the prompt
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_rotate_kernel.pkl']
    
    # Identify the main data file
    outer_path = None
    for path in data_paths:
        if 'standard_data_rotate_kernel.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("Error: 'standard_data_rotate_kernel.pkl' not found in provided paths.")
        sys.exit(1)
        
    # Load the data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data from {outer_path}: {e}")
        sys.exit(1)
        
    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output', None)
    
    print(f"Running test for rotate_kernel with {len(args)} args and {len(kwargs)} kwargs...")
    
    # Execute the function
    try:
        actual_result = rotate_kernel(*args, **kwargs)
    except Exception as e:
        print("Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)
        
    # Verification
    # Note: rotate_kernel returns a numpy array, not a function/closure.
    # If the function returned a callable, the data generation logic implies we would see
    # additional data files for the child function. Since only one file exists,
    # we treat this as a direct input-output test.
    
    passed, msg = recursive_check(expected_result, actual_result)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()