import sys
import os
import dill
import numpy as np
import torch
import traceback
import warnings

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_aux_structures_resample import aux_structures_resample
except ImportError:
    print("Error: Could not import 'aux_structures_resample' from 'agent_aux_structures_resample.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is not in path, though instructions say it is
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                return False, f"Expected numpy array, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            if not np.allclose(expected, actual, equal_nan=True, atol=1e-5):
                return False, "Value mismatch in numpy array"
            return True, "Success"
        if isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False, "Length mismatch"
            for e, a in zip(expected, actual):
                ok, msg = recursive_check(e, a)
                if not ok:
                    return False, msg
            return True, "Success"
        if expected != actual:
            return False, f"Value mismatch: {expected} vs {actual}"
        return True, "Success"

def run_test():
    # 1. DATA FILE ANALYSIS
    # The paths provided in the instructions
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_aux_structures_resample.pkl']
    
    # Identify the outer data file (Function inputs)
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        elif 'standard_data_aux_structures_resample.pkl' in path:
            outer_path = path

    if not outer_path:
        print("Error: Standard data file 'standard_data_aux_structures_resample.pkl' not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")

    # 2. LOAD DATA
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. EXECUTE TARGET FUNCTION
    print("Running 'aux_structures_resample'...")
    try:
        # Based on the function definition, this returns (idx_OUT, Ylm_OUT), not a callable
        # Def: def aux_structures_resample(scheme, lmax=12): ... return idx_OUT, Ylm_OUT
        actual_result = aux_structures_resample(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFICATION STRATEGY
    # Since the function returns data (tuple of arrays) and not a callable, 
    # we simply verify the output against the stored output.
    
    # However, we must handle the hypothetical Scenario B (Factory) logic just in case the provided 
    # data analysis suggests otherwise.
    # In this specific case, based on the provided reference code, aux_structures_resample returns data.
    # So we proceed with direct comparison.
    
    if inner_path:
        print("Warning: Inner data path found but target function appears to return data, not a callable.")
        print("Ignoring inner path for verification.")

    print("Verifying results...")
    passed, msg = recursive_check(expected_outer_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()