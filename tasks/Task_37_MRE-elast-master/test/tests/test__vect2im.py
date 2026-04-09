import sys
import os
import dill
import numpy as np
import traceback

# Conditional import for torch to prevent script crashing if not installed
# The data loader might need it if the pickle contains tensors, but the script entry shouldn't fail.
try:
    import torch
except ImportError:
    torch = None

from agent__vect2im import _vect2im
from verification_utils import recursive_check

def test():
    # Provided data paths
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data__vect2im.pkl']
    
    # Filter for the main function data file
    target_path = None
    for path in data_paths:
        if path.endswith('standard_data__vect2im.pkl'):
            target_path = path
            break
            
    if not target_path:
        print("Skipping test: No matching data file found for _vect2im.")
        sys.exit(0)

    if not os.path.exists(target_path):
        print(f"Skipping test: Data file does not exist at {target_path}")
        sys.exit(0)

    # Load the data
    print(f"Loading data from {target_path}...")
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except ImportError as e:
        if "torch" in str(e):
            print("Failed to load data: The data file contains Torch objects but 'torch' is not installed.")
            sys.exit(1)
        else:
            print(f"Failed to load data: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Running _vect2im with loaded arguments...")

    # Execute the function
    try:
        actual_output = _vect2im(*args, **kwargs)
    except Exception as e:
        print("Execution of _vect2im failed!")
        traceback.print_exc()
        sys.exit(1)

    # Verify the result
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"Verification logic failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test()