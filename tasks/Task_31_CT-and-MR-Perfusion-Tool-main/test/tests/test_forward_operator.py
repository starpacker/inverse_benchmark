import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
# Assuming agent_forward_operator.py is in the Python path or current directory
try:
    from agent_forward_operator import forward_operator
except ImportError:
    # If strictly local in a specific structure, one might need to adjust sys.path
    # But usually, the environment is set up. We'll try a local import fallback if needed.
    sys.path.append(os.getcwd())
    from agent_forward_operator import forward_operator

from verification_utils import recursive_check

def run_test():
    print("Starting Test for: forward_operator")
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Strategy Determination
    # We look for the primary data file for 'forward_operator' and potentially a 'parent_function' file
    # if the operator follows a factory pattern.
    
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if p.endswith('standard_data_forward_operator.pkl'):
            outer_path = p
        elif 'standard_data_parent_function_forward_operator' in p:
            inner_paths.append(p)
            
    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Load Outer Data (Input to forward_operator)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 4. Execute forward_operator
    try:
        print("Executing forward_operator with loaded arguments...")
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification Logic
    # Check if the result is a callable (factory pattern) or a direct result (Scenario A)
    
    if callable(result) and not isinstance(result, (np.ndarray, tuple, list, dict)):
        print("Detected Factory/Closure pattern. Checking for inner data files...")
        
        # Scenario B: Factory Pattern
        if not inner_paths:
            print("Warning: Factory pattern detected but no inner execution data found.")
            # If we expected a callable output, we check if the stored output matches the function object (unlikely to match exactly by value, but strictly checking existence)
            # Usually, in testing factories without inner data, we can't fully verify behavior.
            # However, if 'output' in dill was the function itself, recursive_check might fail on function comparison.
            # Let's check what the recorded output is.
            if callable(expected_outer_output):
                print("Expected output is also callable. Basic existence check passed.")
                sys.exit(0)
            else:
                print("Expected output is data, but we got a callable. Mismatch.")
                sys.exit(1)
        
        # If we have inner paths, we test the closure against them
        all_inner_tests_passed = True
        for i_path in inner_paths:
            try:
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                print(f"Testing inner closure with data from {os.path.basename(i_path)}...")
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output', None)
                
                # Execute the closure
                actual_inner_result = result(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not passed:
                    print(f"Inner test failed for {os.path.basename(i_path)}: {msg}")
                    all_inner_tests_passed = False
                else:
                    print(f"Inner test passed for {os.path.basename(i_path)}.")
                    
            except Exception as e:
                print(f"Error during inner test execution: {e}")
                traceback.print_exc()
                all_inner_tests_passed = False
        
        if all_inner_tests_passed:
            print("All closure tests PASSED.")
            sys.exit(0)
        else:
            print("Some closure tests FAILED.")
            sys.exit(1)

    else:
        # Scenario A: Simple Function (Direct Result)
        print("Detected Direct Result pattern. Verifying output...")
        passed, msg = recursive_check(expected_outer_output, result)
        
        if passed:
            print("Verification Successful: Output matches expected data.")
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"Verification Failed: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()