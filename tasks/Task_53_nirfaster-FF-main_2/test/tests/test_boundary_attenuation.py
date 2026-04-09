import sys
import os
import dill
import numpy as np
import torch
import traceback

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_boundary_attenuation import boundary_attenuation
except ImportError:
    print("Error: Could not import 'boundary_attenuation' from 'agent_boundary_attenuation'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing, though it's expected to be there
    def recursive_check(expected, actual):
        try:
            if isinstance(expected, np.ndarray):
                if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
                    return False, f"Numpy arrays differ. Expected shape {expected.shape}, Actual shape {actual.shape}"
            elif isinstance(expected, (float, int)):
                if abs(expected - actual) > 1e-5:
                    return False, f"Scalars differ: {expected} vs {actual}"
            return True, ""
        except Exception as e:
            return False, str(e)

def run_test():
    # Data paths provided in the prompt
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_boundary_attenuation.pkl']

    # 1. Strategy Analysis
    outer_path = None
    inner_path = None

    for path in data_paths:
        if "standard_data_boundary_attenuation.pkl" in path:
            outer_path = path
        elif "parent_function_boundary_attenuation" in path:
            inner_path = path

    if not outer_path:
        print("Test Skipped: No standard data file found for boundary_attenuation.")
        sys.exit(0)

    print(f"Loading data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # 2. Execution Phase
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing boundary_attenuation with loaded arguments...")
        actual_result = boundary_attenuation(*outer_args, **outer_kwargs)

        # Check if the result is a closure/operator (Scenario B) or a final value (Scenario A)
        # Based on the function definition, boundary_attenuation returns a value (A).
        # However, we handle the closure logic if present for robustness.
        
        final_result = actual_result
        final_expected = expected_output

        if inner_path and callable(actual_result):
            print(f"Detected Closure Pattern. Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                final_expected = inner_data.get('output', None)
                
                print("Executing inner operator...")
                final_result = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Error during inner operator execution: {e}")
                traceback.print_exc()
                sys.exit(1)

        # 3. Verification
        print("Verifying results...")
        passed, msg = recursive_check(final_expected, final_result)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debugging info
            print(f"Expected Type: {type(final_expected)}")
            print(f"Actual Type: {type(final_result)}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()