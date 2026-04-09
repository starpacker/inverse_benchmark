import sys
import os
import dill
import torch
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add the directory containing the target function to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils not found in python path, assuming it's local
    try:
        from verification_utils import recursive_check
    except ImportError:
        print("Warning: verification_utils not found. Defining minimal recursive_check.")
        def recursive_check(expected, actual):
            if isinstance(expected, np.ndarray):
                if not isinstance(actual, np.ndarray):
                    return False, f"Expected numpy array, got {type(actual)}"
                if not np.allclose(expected, actual, equal_nan=True):
                    return False, "Numpy arrays do not match"
                return True, ""
            if expected != actual:
                return False, f"Expected {expected}, got {actual}"
            return True, ""

def test_evaluate_results():
    """
    Unit test for evaluate_results function.
    """
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 1. Identify Data Files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if filename == 'standard_data_evaluate_results.pkl':
            outer_data_path = path
        elif 'standard_data_parent_function_evaluate_results_' in filename:
            inner_data_paths.append(path)

    if not outer_data_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_data_path}...")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Extract inputs and expected outputs
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_result = outer_data.get('output')
    except Exception as e:
        print(f"Error extracting data from pickle: {e}")
        sys.exit(1)

    # 3. Execution
    print("Executing evaluate_results...")
    try:
        # The function saves a plot, let's ensure it doesn't block or fail on display if no display exists
        plt.switch_backend('Agg') 
        
        actual_result = evaluate_results(*args, **kwargs)
        
        # Scenario Check: Did it return a function (Closure/Factory pattern) or a value?
        # Based on the function code provided, it returns 'psnr' (float), so it's a value.
        # However, we must handle the generic logic if inner paths existed (they don't here).
        
        if inner_data_paths:
            # Scenario B: Factory Pattern (Not applicable for this specific function code, but kept for robustness based on instructions)
            if not callable(actual_result):
                print(f"Error: Expected a callable due to existence of inner data files, but got {type(actual_result)}")
                sys.exit(1)
                
            for inner_path in inner_data_paths:
                print(f"  Testing inner execution with {os.path.basename(inner_path)}...")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                inner_actual = actual_result(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(inner_expected, inner_actual)
                if not passed:
                    print(f"FAILED: Inner execution check failed. {msg}")
                    sys.exit(1)
        else:
            # Scenario A: Simple Function
            passed, msg = recursive_check(expected_result, actual_result)
            if not passed:
                print(f"FAILED: Output mismatch. {msg}")
                # Debug info
                print(f"Expected: {expected_result}")
                print(f"Actual:   {actual_result}")
                sys.exit(1)

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_evaluate_results()