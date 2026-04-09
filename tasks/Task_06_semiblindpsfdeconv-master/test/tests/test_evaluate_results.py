import sys
import os
import dill
import traceback
import numpy as np
import torch

# Ensure the current directory is in the path to import local modules
sys.path.append(os.getcwd())

try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback simple check if verification_utils is missing, though recursive_check is expected
    def recursive_check(expected, actual):
        try:
            if isinstance(expected, (np.ndarray, torch.Tensor)):
                expected = np.array(expected)
                actual = np.array(actual)
                if not np.allclose(expected, actual, rtol=1e-3, atol=1e-3):
                    return False, f"Arrays differ. Max diff: {np.max(np.abs(expected - actual))}"
            elif expected != actual:
                return False, f"Values differ: {expected} vs {actual}"
            return True, "OK"
        except Exception as e:
            return False, f"Comparison failed: {e}"

def run_test():
    # 1. DATA FILE ANALYSIS
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. EXECUTION PHASE 1: Run the main function
    print("Executing 'evaluate_results' with outer arguments...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the primary function
        result_phase_1 = evaluate_results(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed in Phase 1: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. EXECUTION PHASE 2: Determine if Factory Pattern or Simple Return
    actual_result = result_phase_1
    expected_result = outer_data.get('output')

    # Scenario B: Factory Pattern (if inner data exists)
    if inner_path:
        print(f"Detected Factory Pattern. Loading inner data from: {inner_path}")
        
        if not callable(result_phase_1):
            print(f"Error: Inner data exists, expecting a callable return from 'evaluate_results', but got {type(result_phase_1)}")
            sys.exit(1)
            
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            print("Executing returned operator (Agent) with inner arguments...")
            actual_result = result_phase_1(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"Execution failed in Phase 2 (Inner Function): {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario A: Simple Function (already handled by default assignments above)
    else:
        print("Scenario A: Simple function execution (No inner data found).")

    # 4. VERIFICATION
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected type: {type(expected_result)}")
        print(f"Actual type: {type(actual_result)}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()