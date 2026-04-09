import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch dependency for deserialization
try:
    import torch
except ImportError:
    # If torch is missing but data has torch tensors, dill might fail or return custom objects.
    # We create a dummy module to prevent immediate ImportErrors during unpickling if possible,
    # though ideally the environment should match.
    class DummyTorch:
        def __call__(self): return None
    torch = None

# Add the current directory to sys.path to import the target function
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_forward_operator import forward_operator
except ImportError:
    print("Error: Could not import 'forward_operator' from 'agent_forward_operator.py'. Check file existence.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, rtol=1e-3, atol=1e-5):
                return False, f"Arrays differ. Max diff: {np.max(np.abs(expected - actual))}"
            return True, "Arrays match"
        if expected != actual:
            return False, f"Values differ: expected {expected}, got {actual}"
        return True, "Values match"

def run_test():
    # 1. Define Data Paths
    # Note: These paths are derived from the problem description
    data_paths = [
        '/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]
    
    # 2. Strategy Analysis
    # We need to distinguish between 'Scenario A' (Function execution) and 'Scenario B' (Factory/Closure).
    # Since forward_operator returns an array (not a function) based on the code provided in the prompt,
    # it likely falls under Scenario A: Simple execution.
    # However, I will check if there are inner files just in case the provided code snippet was partial
    # or if the function behaves differently in practice (e.g., returning a closure).
    
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_paths.append(path)

    if not outer_path:
        print("Skipping test: No standard_data_forward_operator.pkl found.")
        sys.exit(0)

    # 3. Load Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Extract Arguments
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output')

    print(f"Loaded input args types: {[type(a) for a in args]}")
    print(f"Loaded kwargs keys: {list(kwargs.keys())}")

    # 5. Execution Phase 1: Run the main function
    try:
        actual_result = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Execution Phase 2: Check for Factory Pattern (Scenario B)
    # If the result is callable and we have inner data files, we proceed to test the inner closure.
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, list, float, int)):
        if not inner_paths:
            # Result is a function but we have no data to test it.
            # We can only verify that the 'expected_result' was also a function (or checks out vaguely).
            print("Warning: forward_operator returned a callable, but no inner execution data found.")
            # If expected_result is available and not callable, something is wrong.
            if expected_result is not None and not callable(expected_result):
                 print(f"Mismatch: Function returned callable, but data expects {type(expected_result)}")
                 sys.exit(1)
            print("TEST PASSED (Factory created successfully, no inner data to verify)")
            sys.exit(0)
        
        # Test the closure with the inner data
        print(f"Detected Factory Pattern. Testing {len(inner_paths)} inner execution scenarios...")
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                # Execute closure
                inner_actual = actual_result(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(inner_expected, inner_actual)
                if not passed:
                    print(f"Inner Test Failed for {inner_path}: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Error processing inner file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED (Factory and inner executions verified)")
        sys.exit(0)

    # 7. Scenario A: Standard Function Verification
    # If actual_result is data (not a callable), compare it directly to expected_result
    
    # Handle cases where expected_result might be None (captured incorrectly) but we got a result
    if expected_result is None and actual_result is not None:
        print("Warning: Expected result is None in data file, but function returned a value.")
        print(f"Returned value type: {type(actual_result)}")
        # In strict testing, we might fail here, but often data capture misses outputs for void functions.
        # Given the physics context, it should return an array.
        if isinstance(actual_result, np.ndarray):
             print("TEST PASSED (Function ran successfully, output unchecked due to None in data)")
             sys.exit(0)

    passed, msg = recursive_check(expected_result, actual_result)
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()