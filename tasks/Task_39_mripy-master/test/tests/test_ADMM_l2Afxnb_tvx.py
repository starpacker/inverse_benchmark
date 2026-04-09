import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent_ADMM_l2Afxnb_tvx import ADMM_l2Afxnb_tvx
from verification_utils import recursive_check

# ==============================================================================
# HELPER INJECTION (CRITICAL FOR DILL DESERIALIZATION)
# ==============================================================================
# The traceback shows a missing 'dim_match' function inside the serialized objects.
# We must define it in the global scope so the deserialized code can find it.

def dim_match(shape1, shape2):
    """
    Helper function often used in MRI reconstruction operators to handle 
    broadcasting dimensions between sensitivity maps and images.
    """
    if len(shape1) == len(shape2):
        return (shape1, shape2)
    elif len(shape1) > len(shape2):
        shape2_new = shape2 + (1,) * (len(shape1) - len(shape2))
        return (shape1, shape2_new)
    else:
        shape1_new = shape1 + (1,) * (len(shape2) - len(shape1))
        return (shape1_new, shape2)

# Ensure it's available globally for dill
globals()['dim_match'] = dim_match

# ==============================================================================
# TEST LOGIC
# ==============================================================================

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_ADMM_l2Afxnb_tvx.pkl']
    
    # 2. Identify Test Strategy
    # ADMM_l2Afxnb_tvx is typically a direct optimization routine, not a factory.
    # It takes Afunc, invAfunc, b, etc., and returns the reconstructed image 'x'.
    # Therefore, we look for Scenario A (Direct Execution).
    
    target_path = None
    for p in data_paths:
        if p.endswith('standard_data_ADMM_l2Afxnb_tvx.pkl'):
            target_path = p
            break
            
    if not target_path:
        print("Skipping test: No standard_data_ADMM_l2Afxnb_tvx.pkl found.")
        sys.exit(0)

    print(f"Loading data from: {target_path}")
    
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output', None)

    print("Executing ADMM_l2Afxnb_tvx...")
    
    try:
        # 3. Execution
        # Note: The serialized args (Afunc, invAfunc) might rely on global functions 
        # like 'dim_match' which we injected above.
        actual_result = ADMM_l2Afxnb_tvx(*args, **kwargs)
        
    except Exception as e:
        print(f"Error during execution of ADMM_l2Afxnb_tvx: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    is_correct, msg = recursive_check(expected_result, actual_result)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()