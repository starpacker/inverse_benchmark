import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Define global helpers that might be needed during dill unpickling
# based on the provided gen_data_code context.
def _fix_seeds_(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _analyze_obj_(obj):
    pass # Stub, not needed for unpickling usually unless referred

# Inject stubs into global scope if dill looks for them
globals()['_fix_seeds_'] = _fix_seeds_
globals()['_analyze_obj_'] = _analyze_obj_

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if "parent_function_forward_operator" in p:
            inner_paths.append(p)
        elif "standard_data_forward_operator.pkl" in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Extract Outer Args
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output')

    print(f"Executing forward_operator with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
    
    try:
        # 3. Run the Target Function
        actual_result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine Validation Strategy
    # The provided data path list only contains the base function data.
    # Looking at the function source, `forward_operator` returns an array (sinogram), 
    # not a callable/closure. So this is Scenario A.
    
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, torch.Tensor)):
        # Scenario B: Factory Pattern (Closure)
        if not inner_paths:
             print("Error: Function returned a callable (closure), but no inner execution data found.")
             sys.exit(1)
        
        print("Function returned a callable. Proceeding to test inner closure execution.")
        for inner_path in inner_paths:
            print(f"Testing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Failed to load inner data {inner_path}: {e}")
                sys.exit(1)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')
            
            try:
                inner_result = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Error executing inner closure: {e}")
                traceback.print_exc()
                sys.exit(1)
                
            passed, msg = recursive_check(expected_inner_output, inner_result)
            if not passed:
                print(f"Verification FAILED for {inner_path}")
                print(msg)
                sys.exit(1)
    else:
        # Scenario A: Direct Result
        print("Function returned a direct result (non-callable). Verifying against outer output.")
        passed, msg = recursive_check(expected_outer_output, actual_result)
        if not passed:
            print("Verification FAILED")
            print(msg)
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()