import sys
import os
import dill
import numpy as np
import cv2
import traceback

# Add the current directory to sys.path to import the target module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    # Fallback if verification_utils is not in the same dir, define a simple one for robustness
    def recursive_check(expected, actual, tol=1e-5):
        if isinstance(expected, (np.ndarray, list, tuple)):
            if isinstance(expected, (list, tuple)):
                if len(expected) != len(actual):
                    return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
                for i, (e, a) in enumerate(zip(expected, actual)):
                    ok, msg = recursive_check(e, a, tol)
                    if not ok: return False, f"Item {i}: {msg}"
            else: # numpy array
                if expected.shape != actual.shape:
                    return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
                if not np.allclose(expected, actual, atol=tol, rtol=tol):
                    return False, "Array values mismatch"
        return True, "Match"

def test_load_and_preprocess_data():
    """
    Test script for load_and_preprocess_data using pickle data.
    """
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify the main data file
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in path:
            outer_path = path
        elif 'parent_function' in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: Standard data file 'standard_data_load_and_preprocess_data.pkl' not found.")
        sys.exit(1)

    # Load the outer data (inputs for the main function)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output', None)

    print(f"Running test for function: {outer_data.get('func_name')}")
    
    # --- Execution Logic ---
    # Based on the provided code, load_and_preprocess_data is a standard function, 
    # not a factory creating a closure. However, we check if inner paths exist just in case.
    
    actual_result = None

    try:
        # 1. Execute the main function
        # NOTE: The function reads an image from disk. The pickle file contains the arguments (image path).
        # We must ensure that image path exists or simulate the cv2.imread if the environment is isolated.
        # However, the prompt implies running in the environment where paths are valid or inputs are managed.
        # If the path in args doesn't exist, we might fail unless we mock. 
        # Given "robust unit test", we assume the environment matches the recording or we proceed and catch errors.
        
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)

        # 2. Check if the result is a callable (Closure/Factory pattern check)
        # If the function returns another function, we need to process inner_paths.
        if callable(actual_result) and not isinstance(actual_result, (np.ndarray, tuple, list)):
             if not inner_paths:
                 print("Error: Function returned a callable (closure), but no inner data files were provided to test execution.")
                 sys.exit(1)
             
             operator = actual_result
             print("Function returned an operator. Testing inner execution...")
             
             # Test against first available inner data (usually one execution flow)
             inner_path = inner_paths[0]
             with open(inner_path, 'rb') as f:
                 inner_data = dill.load(f)
            
             inner_args = inner_data.get('args', [])
             inner_kwargs = inner_data.get('kwargs', {})
             expected_result = inner_data.get('output') # Override expected result with the result of the closure execution
             
             actual_result = operator(*inner_args, **inner_kwargs)
        
        # If it wasn't a closure, actual_result and expected_result are already set from step 1.

    except FileNotFoundError as fnf:
        print(f"Test Failed: Input file not found during execution. {fnf}")
        sys.exit(1)
    except Exception as e:
        print(f"Test Failed during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Verification ---
    print("Verifying results...")
    try:
        is_match, msg = recursive_check(expected_result, actual_result)
        
        if is_match:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_load_and_preprocess_data()