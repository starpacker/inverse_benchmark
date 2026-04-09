import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_symmetric_gaussian_2d import symmetric_gaussian_2d
from verification_utils import recursive_check

def test_symmetric_gaussian_2d():
    """
    Robust Unit Test for symmetric_gaussian_2d
    """
    # 1. Configuration and Data Paths
    data_dir = '/data/yjh/storm-analysis-master_sandbox/run_code/std_data'
    outer_data_path = os.path.join(data_dir, 'standard_data_symmetric_gaussian_2d.pkl')
    
    # Check if data exists
    if not os.path.exists(outer_data_path):
        print(f"Skipping test: Data file not found at {outer_data_path}")
        sys.exit(0)

    # 2. Load Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded data from {outer_data_path}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Analyze Data Structure (Scenario A vs B)
    # The target function symmetric_gaussian_2d returns an array (g.ravel()), 
    # so it is expected to be Scenario A (Direct Execution). 
    # However, we check if there are inner files just in case.
    
    # Filter for potential inner files (closure executions)
    inner_files = [
        f for f in os.listdir(data_dir) 
        if f.startswith('standard_data_parent_function_symmetric_gaussian_2d_') 
        and f.endswith('.pkl')
    ]

    try:
        if not inner_files:
            # --- Scenario A: Simple Function Execution ---
            print("Mode: Direct Function Execution")
            
            # Extract inputs
            args = outer_data.get('args', [])
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')

            # Execute
            print(f"Executing symmetric_gaussian_2d with {len(args)} args and {len(kwargs)} kwargs...")
            actual_output = symmetric_gaussian_2d(*args, **kwargs)

            # Verify
            passed, msg = recursive_check(expected_output, actual_output)
            
        else:
            # --- Scenario B: Factory/Closure Pattern ---
            # (Unlikely for this specific function, but implementing for robustness)
            print("Mode: Factory/Closure Execution")
            
            # Step 1: Create the operator
            outer_args = outer_data.get('args', [])
            outer_kwargs = outer_data.get('kwargs', {})
            print("Initializing operator...")
            operator = symmetric_gaussian_2d(*outer_args, **outer_kwargs)
            
            if not callable(operator):
                # If it's not callable but inner files exist, something is wrong with our assumption 
                # or the previous data generation. We fall back to treating the first result as the output.
                print("Warning: Result is not callable, reverting to direct comparison despite inner files.")
                expected_output = outer_data.get('output')
                passed, msg = recursive_check(expected_output, operator)
            else:
                # Step 2: Execute the operator with inner data
                # We pick the first inner file available
                inner_path = os.path.join(data_dir, inner_files[0])
                print(f"Loading inner data from {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print("Executing inner operator...")
                actual_output = operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_output, actual_output)

        # 4. Final Result Handling
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_symmetric_gaussian_2d()