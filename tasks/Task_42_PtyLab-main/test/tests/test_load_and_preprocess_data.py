import sys
import os
import dill
import numpy as np
import traceback

# Add specific paths to sys.path to ensure imports work correctly
sys.path.append('/data/yjh/PtyLab-main_sandbox/run_code')

# Import the target function
try:
    from agent_load_and_preprocess_data import load_and_preprocess_data
except ImportError:
    print("Error: Could not import 'load_and_preprocess_data' from 'agent_load_and_preprocess_data.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is not found, though expected in environment
    def recursive_check(expected, actual):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k in expected:
                if k not in actual:
                    return False, f"Key {k} missing in actual output"
                stat, msg = recursive_check(expected[k], actual[k])
                if not stat:
                    return False, f"Key {k}: {msg}"
            return True, "Match"
        if isinstance(expected, (np.ndarray, list)):
            expected_arr = np.array(expected)
            actual_arr = np.array(actual)
            if expected_arr.shape != actual_arr.shape:
                return False, f"Shape mismatch: {expected_arr.shape} vs {actual_arr.shape}"
            # Complex number support
            if np.iscomplexobj(expected_arr) or np.iscomplexobj(actual_arr):
                if not np.allclose(expected_arr, actual_arr, atol=1e-5):
                     return False, "Complex Array content mismatch"
            else:
                if not np.allclose(expected_arr, actual_arr, atol=1e-5):
                    return False, "Array content mismatch"
            return True, "Match"
        if expected != actual:
            return False, f"Value mismatch: {expected} != {actual}"
        return True, "Match"

def test_load_and_preprocess_data():
    """
    Unit test for load_and_preprocess_data.
    Handles both direct execution and factory/closure patterns based on available data files.
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 2. Categorize Paths
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_load_and_preprocess_data' in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: No standard data file found for 'load_and_preprocess_data'.")
        sys.exit(1)

    # 3. Load Outer Data
    print(f"Loading test data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Extract Outer Args
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_result = outer_data.get('output', None)

    # 4. Phase 1: Run the Target Function
    print("Executing 'load_and_preprocess_data' with loaded arguments...")
    try:
        result_phase_1 = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Determine Verification Strategy based on Inner Paths (Closure vs Direct)
    final_result = None
    expected_final_result = None

    if inner_paths and callable(result_phase_1):
        print(f"Detected factory pattern. Testing {len(inner_paths)} inner execution(s)...")
        # In this scenario, we just test the first available inner path for brevity in this unit test
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner pickle file: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_final_result = inner_data.get('output', None)
        
        print("Executing created operator with inner arguments...")
        try:
            final_result = result_phase_1(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Simple function scenario (Scenario A)
        print("Detected direct execution pattern.")
        final_result = result_phase_1
        expected_final_result = expected_outer_result

    # 6. Compare Results
    print("Verifying results...")
    passed, msg = recursive_check(expected_final_result, final_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Optional: inspect keys if dict
        if isinstance(expected_final_result, dict) and isinstance(final_result, dict):
            print(f"Expected keys: {expected_final_result.keys()}")
            print(f"Actual keys:   {final_result.keys()}")
        sys.exit(1)

if __name__ == "__main__":
    test_load_and_preprocess_data()