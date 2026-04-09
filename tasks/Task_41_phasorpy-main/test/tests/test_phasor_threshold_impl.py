import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_phasor_threshold_impl import phasor_threshold_impl
except ImportError:
    print("Error: Could not import 'phasor_threshold_impl' from 'agent_phasor_threshold_impl.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'")
    sys.exit(1)

def run_test():
    # Defined data paths
    data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_phasor_threshold_impl.pkl']
    
    # Filter for the main data file
    target_file = None
    for path in data_paths:
        if 'standard_data_phasor_threshold_impl.pkl' in path:
            target_file = path
            break
            
    if not target_file:
        print("Error: Standard data file 'standard_data_phasor_threshold_impl.pkl' not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {target_file}...")
    
    try:
        with open(target_file, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output', None)

    print(f"Executing phasor_threshold_impl with {len(args)} args and {len(kwargs)} kwargs...")

    try:
        # Execute the function
        actual_result = phasor_threshold_impl(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verify results
    print("Verifying results...")
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