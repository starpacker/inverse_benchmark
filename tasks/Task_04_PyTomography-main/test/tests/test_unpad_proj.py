import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_unpad_proj import unpad_proj
from verification_utils import recursive_check

# Provided data paths
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_unpad_proj.pkl']

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Classify Data Files
    outer_data_path = None
    inner_data_path = None

    for p in data_paths:
        if 'standard_data_unpad_proj.pkl' in p:
            outer_data_path = p
        elif 'standard_data_parent_unpad_proj_' in p: # Matches the naming convention in gen_data_code
            inner_data_path = p
        elif 'standard_data_parent_function_unpad_proj_' in p: # Fallback for alternate naming
            inner_data_path = p

    if not outer_data_path:
        print("Error: Standard input data file (standard_data_unpad_proj.pkl) not found.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_data_path}")
    outer_data = load_data(outer_data_path)
    
    # 2. Execution Phase
    try:
        # Step A: Run the main function
        print("Executing unpad_proj with loaded arguments...")
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Ensure tensors are on the correct device if possible, though CPU is safer for unit tests usually
        # For this specific snippet, unpad_proj takes a Tensor and returns a Tensor directly.
        # It's not a closure returning a function based on the provided source code, 
        # but the decorator logic handles callable results.
        # Based on the provided `unpad_proj` source code:
        # def unpad_proj(proj: torch.Tensor): ... return proj[...]
        # It returns a Tensor, not a function.
        
        result_step_1 = unpad_proj(*outer_args, **outer_kwargs)

        # Step B: Determine if we need a second execution step (Closure pattern) or just verification
        if inner_data_path and callable(result_step_1):
            print(f"Detected closure pattern. Loading inner data from: {inner_data_path}")
            inner_data = load_data(inner_data_path)
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            print("Executing inner operator...")
            actual_result = result_step_1(*inner_args, **inner_kwargs)
            expected_result = inner_data.get('output')
        else:
            # Scenario A: Direct result
            print("Simple function execution detected.")
            actual_result = result_step_1
            expected_result = outer_data.get('output')

    except Exception:
        traceback.print_exc()
        print("Error during test execution.")
        sys.exit(1)

    # 3. Verification Phase
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()