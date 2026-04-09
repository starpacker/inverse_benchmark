import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
from agent__bs_three_point import _bs_three_point
from verification_utils import recursive_check

def test_bs_three_point():
    """
    Unit test for _bs_three_point using captured standard data.
    """
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data__bs_three_point.pkl']
    
    # Identify the primary data file
    outer_path = None
    for path in data_paths:
        if path.endswith('standard_data__bs_three_point.pkl'):
            outer_path = path
            break
            
    if not outer_path:
        print("Error: standard_data__bs_three_point.pkl not found in provided paths.")
        sys.exit(1)

    try:
        # Load the data
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
        
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        print(f"Running _bs_three_point with captured arguments from {outer_path}...")
        
        # Execute the function
        actual_output = _bs_three_point(*args, **kwargs)
        
        # Verify results
        print("Verifying results...")
        is_match, msg = recursive_check(expected_output, actual_output)
        
        if is_match:
            print("TEST PASSED: Output matches expected standard data.")
            sys.exit(0)
        else:
            print(f"TEST FAILED: Output mismatch.\n{msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"TEST FAILED: An execution error occurred.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_bs_three_point()