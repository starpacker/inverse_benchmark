import sys
import os
import dill
import torch
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Ensure strict reproducibility for any random operations
def _fix_seeds(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_fix_seeds(42)

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# Import target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils'")
    sys.exit(1)

def main():
    # Data paths provided in instructions
    data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Identify Data Files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if "standard_data_evaluate_results.pkl" in p:
            outer_path = p
        elif "parent_function_evaluate_results" in p:
            inner_paths.append(p)
    
    if not outer_path:
        print("Error: No 'standard_data_evaluate_results.pkl' found.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Prepare inputs
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # --- Execution Phase ---
    print("\nRunning 'evaluate_results'...")
    try:
        # Scenario A: evaluate_results is a standard function execution (most likely based on provided code)
        # Scenario B: evaluate_results returns a closure (less likely given the function body, but handled logically)
        
        actual_result = evaluate_results(*outer_args, **outer_kwargs)
        
        # If there are inner paths, it means the result of the first call was a callable (Scenario B)
        # But based on the provided data list, there are no inner paths.
        if inner_paths:
            print(f"Detected {len(inner_paths)} inner data files. Assuming Factory Pattern.")
            if not callable(actual_result):
                 print("Error: Expected 'evaluate_results' to return a callable for inner execution, but got:", type(actual_result))
                 sys.exit(1)
            
            # For each inner path, execute the returned operator
            for i_path in inner_paths:
                print(f"  Testing Inner Data: {i_path}")
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                i_expected = inner_data.get('output')
                
                i_actual = actual_result(*i_args, **i_kwargs)
                
                passed, msg = recursive_check(i_expected, i_actual)
                if not passed:
                    print(f"  FAILED inner test at {i_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
            
            print("All inner tests passed.")
            sys.exit(0)

        else:
            # Scenario A: Direct comparison
            print("No inner data files found. Comparing direct output.")
            passed, msg = recursive_check(expected_output, actual_result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("TEST FAILED")
                print(f"Mismatch Details: {msg}")
                sys.exit(1)

    except Exception as e:
        print("Execution Failed with Exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()