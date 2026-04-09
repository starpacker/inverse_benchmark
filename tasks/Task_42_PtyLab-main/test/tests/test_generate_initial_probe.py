import sys
import os
import dill
import numpy as np
import traceback

# Add project root to sys.path to ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_generate_initial_probe import generate_initial_probe
from verification_utils import recursive_check

def test_generate_initial_probe():
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_generate_initial_probe.pkl']
    
    # Identify data file
    outer_data_path = None
    for path in data_paths:
        if path.endswith('standard_data_generate_initial_probe.pkl'):
            outer_data_path = path
            break
            
    if not outer_data_path:
        print("Error: Standard data file not found.")
        sys.exit(1)

    # Load input arguments and expected output
    try:
        with open(outer_data_path, 'rb') as f:
            data = dill.load(f)
            
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_result = data.get('output', None)
        
        print(f"Loaded data from {outer_data_path}")
        print(f"Args: {[type(a) for a in args]}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Execute the function
    try:
        print("Executing generate_initial_probe...")
        actual_result = generate_initial_probe(*args, **kwargs)
    except Exception as e:
        print(f"Error executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verification
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Optional: Detailed debug info for array mismatches
            if isinstance(expected_result, np.ndarray) and isinstance(actual_result, np.ndarray):
                print(f"Expected shape: {expected_result.shape}, Actual shape: {actual_result.shape}")
                print(f"Expected dtype: {expected_result.dtype}, Actual dtype: {actual_result.dtype}")
                if expected_result.shape == actual_result.shape:
                    diff = np.abs(expected_result - actual_result)
                    print(f"Max difference: {np.max(diff)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_generate_initial_probe()