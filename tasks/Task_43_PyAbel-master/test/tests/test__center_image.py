import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent to the path so we can import it
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent__center_image import _center_image
from verification_utils import recursive_check

def test_center_image():
    # 1. Define data paths
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data__center_image.pkl']
    
    # 2. Identify the main data file
    # Based on the provided path, we are in Scenario A: Simple Function execution.
    # The function _center_image returns a numpy array directly, not a closure.
    data_path = data_paths[0]
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # 3. Load the data
    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data with dill: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Loaded data for function: {data.get('func_name', 'unknown')}")

    # 4. Run the function
    try:
        actual_output = _center_image(*args, **kwargs)
    except Exception as e:
        print(f"Error executing _center_image: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify results
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
    test_center_image()