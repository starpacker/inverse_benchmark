import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path so we can import the agent
sys.path.append(os.getcwd())

try:
    from agent_fbp_reconstruct import fbp_reconstruct
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Helper for handling specific serialization contexts if necessary
def _fix_seeds_(seed=42):
    import random
    if np:
        np.random.seed(seed)
    random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    _fix_seeds_(42)

    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_fbp_reconstruct.pkl']
    
    # Identify Data Files
    outer_path = None
    inner_path = None

    for p in data_paths:
        if "parent_function_fbp_reconstruct" in p:
            inner_path = p
        elif "standard_data_fbp_reconstruct.pkl" in p:
            outer_path = p

    if outer_path is None:
        print("Test Skipped: No outer data file found (standard_data_fbp_reconstruct.pkl).")
        sys.exit(0)

    try:
        # Load Outer Data
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output', None)

        print(f"Running fbp_reconstruct with outer args...")
        
        # Execute Outer Function
        # This function might return a result directly (Scenario A) or a closure (Scenario B)
        actual_result_or_operator = fbp_reconstruct(*outer_args, **outer_kwargs)

        # Check for Inner Data (Scenario B: Closure)
        if inner_path:
            print(f"Inner data found at {inner_path}. Treating result as operator/closure.")
            if not callable(actual_result_or_operator):
                print("Error: Expected a callable closure based on file structure, but got a static result.")
                sys.exit(1)
            
            with open(inner_path, "rb") as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)
            
            print("Executing closure with inner args...")
            final_result = actual_result_or_operator(*inner_args, **inner_kwargs)
            expected_final = expected_inner_output
        else:
            # Scenario A: Simple Function
            print("No inner data found. Treating result as final output.")
            final_result = actual_result_or_operator
            expected_final = expected_outer_output

        # Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_final, final_result)
        
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