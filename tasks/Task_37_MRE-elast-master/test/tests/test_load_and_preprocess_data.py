import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the target function to the path
sys.path.append('/data/yjh/MRE-elast-master_sandbox/run_code')

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
    # Fallback if verification_utils is not in python path, define a simple version
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                return False, f"Type mismatch: Expected numpy array, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: Expected {expected.shape}, got {actual.shape}"
            if not np.allclose(expected, actual, equal_nan=True, atol=1e-5):
                return False, f"Value mismatch in numpy array. Max diff: {np.max(np.abs(expected - actual))}"
            return True, "Arrays match"
        
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)):
                return False, f"Type mismatch: Expected {type(expected)}, got {type(actual)}"
            if len(expected) != len(actual):
                return False, f"Length mismatch: Expected {len(expected)}, got {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                ok, msg = recursive_check(e, a)
                if not ok:
                    return False, f"Index {i}: {msg}"
            return True, "Sequence matches"
            
        if expected != actual:
            return False, f"Value mismatch: Expected {expected}, got {actual}"
        
        return True, "Values match"

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    outer_path = None
    inner_path = None

    # 2. Path Analysis
    for path in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in path:
            outer_path = path
        elif 'parent_function_load_and_preprocess_data' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Execution Phase
    try:
        # Load Outer Data
        print(f"Loading outer data from {outer_path}...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute Outer Function
        print("Executing load_and_preprocess_data with outer arguments...")
        try:
            actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # 4. Verification Logic (Scenario A vs B)
        if inner_path:
            # Scenario B: Factory Pattern (Function returns a function)
            if not callable(actual_result):
                print(f"Error: Expected a callable (closure) from outer execution, but got {type(actual_result)}")
                sys.exit(1)
            
            print(f"Loading inner data from {inner_path}...")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            print("Executing inner closure...")
            actual_final_result = actual_result(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function (Direct result comparison)
            print("No inner path found. Comparing direct output.")
            expected_result = outer_data.get('output')
            actual_final_result = actual_result

        # 5. Final Comparison
        print("Verifying results...")
        passed, msg = recursive_check(expected_result, actual_final_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()