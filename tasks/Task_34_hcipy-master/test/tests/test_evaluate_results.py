import sys
import os
import dill
import numpy as np
import traceback
import glob

# CONDITIONAL IMPORT: Handle torch absence gracefully
try:
    import torch
except ImportError:
    torch = None

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def test_evaluate_results():
    # 1. DATA FILE DISCOVERY
    # Paths provided in the prompt context
    base_data_dir = '/data/yjh/hcipy-master_sandbox/run_code/std_data'
    
    # Define file patterns
    outer_data_path = os.path.join(base_data_dir, 'standard_data_evaluate_results.pkl')
    
    # Look for potential inner data files (indicating a factory pattern)
    # Pattern: standard_data_parent_function_evaluate_results_*.pkl
    inner_data_pattern = os.path.join(base_data_dir, 'standard_data_parent_function_evaluate_results_*.pkl')
    inner_data_files = glob.glob(inner_data_pattern)

    if not os.path.exists(outer_data_path):
        print(f"Skipping test: Outer data file not found at {outer_data_path}")
        # If no data exists, we can't test, but it's not strictly a code failure.
        sys.exit(0)

    print(f"Loading outer data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. EXECUTION STRATEGY
    # Check if we have inner files. 
    # If inner files exist, 'evaluate_results' is likely a factory returning a callable.
    # If no inner files, 'evaluate_results' is a standard function.
    
    is_factory_mode = len(inner_data_files) > 0
    
    try:
        # --- PHASE 1: EXECUTE MAIN FUNCTION ---
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        print("Executing evaluate_results with outer arguments...")
        # We catch plotting warnings/errors specifically if environment lacks display
        try:
            primary_result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            # If it's just a display error (common in headless CI), we might warn but proceed if we expected None
            print(f"Execution error in evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        # --- PHASE 2: HANDLE RESULTS ---
        if is_factory_mode:
            print(f"Factory mode detected. {len(inner_data_files)} inner execution files found.")
            
            if not callable(primary_result):
                print(f"Error: Expected evaluate_results to return a callable in factory mode, got {type(primary_result)}")
                sys.exit(1)

            # Iterate through inner data files to test the closure/operator
            for inner_path in inner_data_files:
                print(f"Testing inner execution with: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')

                # Execute the agent (operator)
                actual_inner_output = primary_result(*inner_args, **inner_kwargs)

                # Verify
                passed, msg = recursive_check(expected_inner_output, actual_inner_output)
                if not passed:
                    print(f"FAILED: Inner execution mismatch for {inner_path}")
                    print(msg)
                    sys.exit(1)
                else:
                    print(f"Inner execution passed for {inner_path}")

        else:
            # Scenario A: Simple Function
            print("Standard function mode detected.")
            expected_output = outer_data.get('output')
            
            # Since evaluate_results produces plots/files, the return value is likely None.
            # We verify that the return value matches the recorded return value.
            passed, msg = recursive_check(expected_output, primary_result)
            if not passed:
                print("FAILED: Output mismatch.")
                print(f"Expected: {expected_output}")
                print(f"Actual: {primary_result}")
                print(msg)
                sys.exit(1)
            else:
                print("Output matches expected (likely None for void functions).")

    except Exception as e:
        print(f"An unexpected error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Clean up generated image if it exists to keep workspace clean
    if os.path.exists("hcipy_standalone_results.png"):
        try:
            os.remove("hcipy_standalone_results.png")
            print("Cleaned up generated artifact: hcipy_standalone_results.png")
        except:
            pass

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_evaluate_results()