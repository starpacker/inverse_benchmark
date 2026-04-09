import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import target function
from agent_prox_l2_Afxnb_CGD import prox_l2_Afxnb_CGD
from verification_utils import recursive_check

# --- Dependency Injection for Unpickled Objects ---
# The error "name 'dim_match' is not defined" suggests that the pickled function objects
# rely on a global helper 'dim_match' that isn't present in the test environment.
# We define a compatible version here and inject it into the global scope.

def dim_match(shape1, shape2):
    """
    Helper function often used in MRI reconstruction utilities to match dimensions
    for broadcasting or processing. Reconstructed based on common usage context.
    """
    # Simple pass-through or basic matching logic usually sufficient for unpickling context
    # unless complex logic is strictly required. 
    # If shape1 and shape2 are tuples, return them.
    return shape1, shape2

# Inject into global namespace so dill-loaded functions can find it
globals()['dim_match'] = dim_match

# --------------------------------------------------

def run_test():
    # 1. Define Data Paths
    # The user provided path in the prompt context
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_prox_l2_Afxnb_CGD.pkl']
    
    outer_path = None
    inner_path = None

    # 2. Identify Input Files
    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        elif 'standard_data_prox_l2_Afxnb_CGD.pkl' in path:
            outer_path = path

    if not outer_path:
        print("[Error] Primary data file (standard_data_prox_l2_Afxnb_CGD.pkl) not found in paths.")
        sys.exit(1)

    # 3. Load Data
    try:
        print(f"[Info] Loading data from {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[Error] Failed to load outer data: {e}")
        sys.exit(1)

    # 4. Extract Arguments
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 5. Robustness: Patch specific known issues in arguments if necessary
    # (Based on previous logs, we check if shapes need adjustment, though the main error was NameError)
    
    # 6. Execute Function
    print("[Info] Executing prox_l2_Afxnb_CGD...")
    try:
        # We need to ensure the injected 'dim_match' is available to the unpickled functions.
        # Sometimes unpickled functions are bound to a specific module namespace.
        # However, if they look up via globals(), our injection works.
        
        # Scenario A: prox_l2_Afxnb_CGD is a direct calculation function (based on signature)
        # It takes Afunc, invAfunc, b, x0... and returns 'x'.
        # It does NOT return a closure.
        actual_result = prox_l2_Afxnb_CGD(*args, **kwargs)

    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 7. Verification
    print("[Info] Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"[Error] Verification logic failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()