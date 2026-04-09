import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the agent code
sys.path.append(os.getcwd())

try:
    from agent_calculate_psnr import calculate_psnr
except ImportError:
    print("Error: Could not import 'calculate_psnr' from 'agent_calculate_psnr.py'.")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def run_test():
    # 1. Define Data Paths
    outer_path = '/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_calculate_psnr.pkl'
    
    # 2. Check for existence
    if not os.path.exists(outer_path):
        print(f"Error: Data file not found at {outer_path}")
        sys.exit(1)

    # 3. Load Data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Loaded data from {outer_path}")
    print(f"Function: {data.get('func_name')}")
    # Basic analysis of inputs for debugging context
    for i, arg in enumerate(args):
        if hasattr(arg, 'shape'):
            print(f"Arg {i} shape: {arg.shape}, type: {type(arg)}")
        else:
            print(f"Arg {i} type: {type(arg)}")

    # 4. Execute the function
    try:
        actual_output = calculate_psnr(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify Results
    passed, msg = recursive_check(expected_output, actual_output)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()