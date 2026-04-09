import sys
import os
import dill
import numpy as np
import warnings
import traceback
import torch

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent__scheme2noddi import _scheme2noddi
except ImportError:
    print("CRITICAL: Could not import '_scheme2noddi' from 'agent__scheme2noddi.py'. Make sure the file exists.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing, mainly for standalone robustness
    def recursive_check(expected, actual):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k in expected:
                if k not in actual:
                    return False, f"Key {k} missing in actual"
                ok, msg = recursive_check(expected[k], actual[k])
                if not ok:
                    return False, f"Key {k}: {msg}"
            return True, ""
        if isinstance(expected, (np.ndarray, list)):
            expected = np.array(expected)
            actual = np.array(actual)
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
            # Handle string arrays or object arrays which can't do allclose
            if expected.dtype.kind in {'U', 'S', 'O'}:
                if not np.array_equal(expected, actual):
                    return False, "String/Object array mismatch"
                return True, ""
            if not np.allclose(expected, actual, equal_nan=True):
                return False, f"Value mismatch. Max diff: {np.max(np.abs(expected - actual))}"
            return True, ""
        if expected != actual:
            return False, f"Value mismatch: {expected} != {actual}"
        return True, ""

def test_scheme2noddi():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data__scheme2noddi.pkl']
    
    # 2. Identify the Outer Data File (Scenario A or B start)
    outer_path = None
    for path in data_paths:
        if 'standard_data__scheme2noddi.pkl' in path:
            outer_path = path
            break
            
    if not outer_path or not os.path.exists(outer_path):
        print(f"SKIPPED: Data file not found at {outer_path}")
        # If data is missing, we can't test, but we shouldn't fail the CI pipeline if this is expected.
        # However, for a unit test script designed to run, we usually exit 1 if setup fails.
        # Assuming strict requirement:
        sys.exit(1)

    print(f"Loading data from: {outer_path}")
    
    # 3. Load Data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load pickle file. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Extract Arguments and Expected Output
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
    except Exception as e:
        print(f"CRITICAL: Corrupt data structure in pickle. Error: {e}")
        sys.exit(1)

    # 5. Run the Target Function
    print("Running _scheme2noddi...")
    try:
        # Based on analysis, _scheme2noddi returns a dictionary (protocol), not a callable.
        # This is Scenario A.
        actual_output = _scheme2noddi(*args, **kwargs)
    except Exception as e:
        print(f"CRITICAL: Execution of _scheme2noddi failed. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Verify Results
    print("Verifying results...")
    is_correct, fail_msg = recursive_check(expected_output, actual_output)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {fail_msg}")
        # Debugging aid: print keys if it's a dict mismatch
        if isinstance(expected_output, dict) and isinstance(actual_output, dict):
            print(f"Expected keys: {list(expected_output.keys())}")
            print(f"Actual keys:   {list(actual_output.keys())}")
        sys.exit(1)

if __name__ == "__main__":
    test_scheme2noddi()