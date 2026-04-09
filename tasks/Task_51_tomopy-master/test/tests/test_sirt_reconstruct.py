import sys
import os
import dill
import numpy as np
import scipy.fft
import scipy.ndimage
import torch
import traceback

# Add current directory to path so we can import the agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_sirt_reconstruct import sirt_reconstruct
except ImportError:
    print("Could not import sirt_reconstruct from agent_sirt_reconstruct. Check file structure.")
    sys.exit(1)

from verification_utils import recursive_check

# --- Helpers for pickle loading ---
# We need to make sure the environment matches what dill expects if it serialized
# objects dependent on global imports.
# The provided gen_data_code uses scipy.ndimage, scipy.fft, numpy, etc.

def load_pickle_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return dill.load(f)

def main():
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_sirt_reconstruct.pkl']
    
    # 1. Identify File Types
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_sirt_reconstruct.pkl' in p:
            outer_path = p
        elif 'parent_function_sirt_reconstruct' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Test Skipped: No standard_data_sirt_reconstruct.pkl found.")
        sys.exit(0)

    # 2. Load Data
    try:
        outer_data = load_pickle_data(outer_path)
    except Exception as e:
        print(f"Failed to load pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not outer_data:
        print("Error: Pickle data was empty or failed to load.")
        sys.exit(1)

    print(f"Loaded outer data from {outer_path}")
    
    # 3. Execution Strategy
    # sirt_reconstruct in the reference code is a direct function returning a reconstruction (numpy array).
    # It is NOT a factory function returning a callable.
    # Therefore, we strictly follow Scenario A (Simple Function).

    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Running sirt_reconstruct with {len(args)} args and {len(kwargs)} kwargs...")

    try:
        # Execute the agent function
        actual_result = sirt_reconstruct(*args, **kwargs)
    except Exception as e:
        print(f"Execution of sirt_reconstruct failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    # If the function returns a callable (Scenario B - Factory), we would need to handle inner paths.
    # However, based on the provided code, it returns a reconstruction array.
    
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, torch.Tensor)):
        # Unexpected scenario given the reference code, but robust handling:
        # If it returned a function, we would look for inner paths to test that function.
        if not inner_paths:
            print("The function returned a callable, but no inner data files were provided to test it.")
            sys.exit(1)
        
        print("Detected Factory Pattern (unexpected but handling). Testing returned operator against inner files...")
        operator = actual_result
        
        for inner_p in inner_paths:
            print(f"  Testing inner file: {inner_p}")
            inner_data = load_pickle_data(inner_p)
            i_args = inner_data.get('args', [])
            i_kwargs = inner_data.get('kwargs', {})
            i_expected = inner_data.get('output')
            
            try:
                i_actual = operator(*i_args, **i_kwargs)
            except Exception as e:
                print(f"  Inner execution failed: {e}")
                sys.exit(1)
                
            passed, msg = recursive_check(i_expected, i_actual)
            if not passed:
                print(f"  FAILED inner comparison: {msg}")
                sys.exit(1)
                
        print("TEST PASSED (Factory Mode)")
        sys.exit(0)

    else:
        # Scenario A: Direct comparison
        print("Verifying direct output...")
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    main()