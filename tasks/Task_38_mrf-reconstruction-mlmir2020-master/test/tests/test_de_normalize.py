import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_de_normalize import de_normalize
from verification_utils import recursive_check

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_de_normalize.pkl']
    
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif 'standard_data_de_normalize.pkl' in p:
            outer_path = p

    if outer_path is None:
        print("Error: standard_data_de_normalize.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}")
    
    # 2. Load Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Logic
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output')

        print("Executing de_normalize with loaded arguments...")
        result = de_normalize(*outer_args, **outer_kwargs)

        # Check if the result is a closure/operator (Scenario B) or a direct value (Scenario A)
        if callable(result) and not isinstance(result, (torch.Tensor, np.ndarray)):
            # This looks like Scenario B (closure), but based on the provided function signature 
            # and the lack of 'parent_function' files in the provided path list,
            # it is likely a direct computation (Scenario A).
            # However, we must handle the possibility if inner files existed.
            
            if not inner_paths:
                # If it's callable but we have no inner data to test it with, we check if the expected output matches the callable object itself
                # or if the function was expected to return a value.
                # In this specific case, de_normalize returns a numpy array, so it shouldn't be a closure.
                # If it IS a closure here, something might be wrong with the assumption or the function definition changed.
                # We proceed to verification assuming the output IS the result.
                pass
            else:
                 # Scenario B Logic (Closure)
                agent_operator = result
                print("Function returned a callable. Proceeding to test inner executions...")
                
                for in_path in inner_paths:
                    with open(in_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', [])
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected_inner_output = inner_data.get('output')
                    
                    print(f"  Testing inner call from {os.path.basename(in_path)}...")
                    inner_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    passed, msg = recursive_check(expected_inner_output, inner_result)
                    if not passed:
                        print(f"  FAILED: Inner execution mismatch in {os.path.basename(in_path)}")
                        print(msg)
                        sys.exit(1)
                
                print("All inner tests passed.")
                sys.exit(0)

        # Scenario A Logic (Direct Result)
        print("Verifying direct output...")
        passed, msg = recursive_check(expected_outer_output, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED: Output mismatch")
            print(msg)
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()