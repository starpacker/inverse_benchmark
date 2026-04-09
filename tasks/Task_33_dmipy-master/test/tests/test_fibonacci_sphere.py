import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the agent code
sys.path.append(os.getcwd())

from agent_fibonacci_sphere import fibonacci_sphere
from verification_utils import recursive_check

def run_test():
    data_paths = ['/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_fibonacci_sphere.pkl']
    
    # 1. Identify the primary data file
    # In this case, there is only one file, which corresponds to the direct execution of the function.
    outer_path = None
    for path in data_paths:
        if 'standard_data_fibonacci_sphere.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("Error: standard_data_fibonacci_sphere.pkl not found in data_paths.")
        sys.exit(1)

    # 2. Load the input data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
        
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output')
        
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execute the function
    try:
        print(f"Running fibonacci_sphere with args={args} kwargs={kwargs}")
        actual_output = fibonacci_sphere(*args, **kwargs)
    except Exception as e:
        print(f"Error executing fibonacci_sphere: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify results
    # Since fibonacci_sphere returns a numpy array directly (not a closure),
    # we verify the output directly against the expected output.
    try:
        passed, msg = recursive_check(expected_output, actual_output)
        
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