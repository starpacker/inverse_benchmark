import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing (basic check)
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, equal_nan=True):
                return False, f"Arrays differ. Expected shape {expected.shape}, got {actual.shape}"
            return True, ""
        if expected != actual:
            return False, f"Expected {expected}, got {actual}"
        return True, ""

def test_evaluate_results():
    """
    Test script for evaluate_results using captured dill data.
    """
    
    # 1. Configuration
    data_paths = ['/data/yjh/PySMLFM-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze paths to distinguish between simple function calls and factory/closure patterns
    outer_data_path = None
    inner_data_paths = []
    
    target_func_name = "evaluate_results"
    
    for path in data_paths:
        if f"standard_data_{target_func_name}.pkl" in path:
            outer_data_path = path
        elif f"standard_data_parent_function_{target_func_name}_" in path:
            inner_data_paths.append(path)

    if not outer_data_path:
        print(f"Error: Standard data file for '{target_func_name}' not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Reconstruct / Execute
    # The 'evaluate_results' function in the provided code is a standard function 
    # that calculates RMSE, not a factory returning another function.
    # However, we must handle the generic logic if it *were* a factory based on the file presence.
    
    try:
        print("Executing 'evaluate_results' with captured arguments...")
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Run the function
        actual_result = evaluate_results(*args, **kwargs)
        
        # 3. Check for Closure/Factory Pattern
        # If inner data files exist, it means the result of the outer function was a callable (the 'agent')
        # that needs to be tested against those inner files.
        if inner_data_paths:
            print(f"Detected {len(inner_data_paths)} inner execution data files. Treating result as an operator.")
            
            if not callable(actual_result):
                print("Error: Inner data exists, but the result of the outer function is not callable.")
                sys.exit(1)
                
            operator = actual_result
            
            for inner_path in inner_data_paths:
                print(f"  - Testing against inner data: {os.path.basename(inner_path)}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                inner_actual_output = operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_inner_output, inner_actual_output)
                if not passed:
                    print(f"FAILED: Inner execution mismatch in {os.path.basename(inner_path)}")
                    print(f"Details: {msg}")
                    sys.exit(1)

            print("All inner executions passed.")

        else:
            # Standard Scenario: The function just returns a value (RMSE float in this case)
            print("No inner data files found. Verifying direct output...")
            expected_output = outer_data.get('output')
            
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print("FAILED: Output mismatch.")
                print(f"Expected: {expected_output}")
                print(f"Actual:   {actual_result}")
                print(f"Details:  {msg}")
                sys.exit(1)
            
            print("Direct output verification successful.")

    except Exception as e:
        print(f"An error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_evaluate_results()