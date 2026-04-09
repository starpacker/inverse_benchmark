import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_reduced_deflection_angle_sie import reduced_deflection_angle_sie

# Import verification utils
from verification_utils import recursive_check

def run_test():
    # 1. Setup Data Paths
    data_paths = [
        '/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_reduced_deflection_angle_sie.pkl'
    ]
    
    # Identify Outer and Inner data files based on the decorator logic
    # The decorator saves inner functions as 'standard_data_parent_function_{parent_name}_{func_name}.pkl'
    outer_path = None
    inner_paths = []

    for path in data_paths:
        filename = os.path.basename(path)
        if "parent_function" in filename:
            inner_paths.append(path)
        elif "reduced_deflection_angle_sie.pkl" in filename:
            outer_path = path

    if not outer_path:
        print("Error: standard_data_reduced_deflection_angle_sie.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. Execute Target Function
    try:
        # Move tensors to GPU if available, matching the likely environment of generation
        if torch.cuda.is_available():
            outer_args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in outer_args]
            outer_kwargs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in outer_kwargs.items()}

        print("Executing reduced_deflection_angle_sie with loaded arguments...")
        actual_result = reduced_deflection_angle_sie(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Execution failed!")
        traceback.print_exc()
        sys.exit(1)

    # 4. Handle Closure vs Direct Output
    # The prompt implies a scenario where it might return an operator (Factory/Closure) 
    # OR a direct result. Based on the provided code, `reduced_deflection_angle_sie` returns 
    # a tuple of tensors (direct result), not a callable. 
    # However, I will check if `inner_paths` exist just in case the provided code snippet 
    # in the prompt differs from the actual execution context (though unlikely given the snippet).
    
    if inner_paths and callable(actual_result):
        # Scenario B: Function returned a closure/operator
        print("Function returned a callable. Testing inner execution...")
        operator = actual_result
        
        for inner_path in inner_paths:
            print(f"  Testing inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output', None)

                # Move inner args to GPU if needed
                if torch.cuda.is_available():
                    inner_args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in inner_args]
                    inner_kwargs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inner_kwargs.items()}

                actual_inner_result = operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not passed:
                    print(f"  Inner verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("  Inner verification passed.")

            except Exception as e:
                print(f"  Inner execution failed for {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Function returned a value (Direct calculation)
        # Based on the reference code provided, this is the expected path.
        print("Function returned a value. verifying result...")
        
        passed, msg = recursive_check(expected_outer_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"Verification Failed: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()