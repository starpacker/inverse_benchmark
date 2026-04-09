import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing agent_precompute_H_fft.py to the system path
# assuming it is in the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_precompute_H_fft import precompute_H_fft
    from verification_utils import recursive_check
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")
    sys.exit(1)

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_precompute_H_fft.pkl']
    
    # 2. Identify the main data file
    # In this case, there seems to be only one file provided in the prompt's list, 
    # which suggests a direct execution test (Scenario A), but we handle Scenario B logic just in case
    # valid inner files are found in a real directory traversal (though here we stick to the provided list).
    
    outer_file = None
    inner_file = None
    
    for path in data_paths:
        if 'standard_data_precompute_H_fft.pkl' in path:
            outer_file = path
        elif 'standard_data_parent_function_precompute_H_fft' in path:
            inner_file = path
            
    if not outer_file:
        print("[ERROR] Standard data file for 'precompute_H_fft' not found in provided paths.")
        sys.exit(1)

    print(f"[INFO] Loading data from {outer_file}...")
    
    # 3. Load Data
    try:
        with open(outer_file, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    # 4. Execute the Target Function
    print("[INFO] Executing 'precompute_H_fft' with loaded arguments...")
    try:
        # Note: If this function creates a closure, 'actual_result' will be a callable.
        # If it returns a value (like an array), it will be the final result.
        actual_result = precompute_H_fft(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Handle Closure/Factory Pattern if applicable
    # Check if the result is callable and if we have inner data to test that callable
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, list, tuple, dict)):
        if inner_file:
            print(f"[INFO] Result is a callable. Loading inner data from {inner_file} to verify closure...")
            try:
                with open(inner_file, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                # Execute the closure
                actual_result = actual_result(*inner_args, **inner_kwargs)
                expected_output = expected_inner_output # Update expectation target
            except Exception as e:
                print(f"[ERROR] Failed to execute inner closure: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("[WARN] Result is callable (Closure detected), but no inner data file provided to test execution.")
            print("[WARN] Validating the closure object itself against expected output (likely the function object).")

    # 6. Verification
    print("[INFO] Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("[SUCCESS] TEST PASSED: Output matches expected data.")
            sys.exit(0)
        else:
            print(f"[FAILURE] TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()