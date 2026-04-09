import sys
import os
import dill
import numpy as np
import traceback
import torch

# Import the target function and class
from agent_create_high_resolution_scheme import create_high_resolution_scheme, Scheme

# Import verification utility
from verification_utils import recursive_check

# --- Data Injection for Dill ---
# The data was likely generated in a context where 'Scheme' was defined in __main__
# or a specific module context. We need to ensure dill can load it.
# We also need to handle the mismatch between the loaded 'Scheme' class and the 
# imported 'Scheme' class.

def reconcile_scheme_objects(obj):
    """
    Recursively traverse the object. If a 'Scheme' object is found that isn't 
    an instance of the imported agent_create_high_resolution_scheme.Scheme,
    convert it or its attributes to be comparable.
    
    Since we cannot easily change the class of the loaded object if it's a closed reference,
    we will rely on comparing __dict__ for Scheme objects in our custom check if types mismatch 
    but names match.
    """
    if isinstance(obj, list):
        return [reconcile_scheme_objects(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(reconcile_scheme_objects(x) for x in obj)
    if isinstance(obj, dict):
        return {k: reconcile_scheme_objects(v) for k, v in obj.items()}
    
    # If it looks like a Scheme but from a different module (e.g. __main__.Scheme)
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Scheme':
        # If it's not exactly the class we imported
        if obj.__class__ is not Scheme:
            # Create a new instance of the correct Scheme class
            # We bypass __init__ to avoid file loading logic since we are copying state
            new_obj = Scheme.__new__(Scheme)
            # Copy all attributes
            new_obj.__dict__ = reconcile_scheme_objects(obj.__dict__)
            return new_obj
            
    return obj

def run_test():
    # Data paths provided in instruction
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_create_high_resolution_scheme.pkl']
    
    outer_path = None
    inner_path = None

    # Identify files
    for path in data_paths:
        if 'standard_data_create_high_resolution_scheme.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_create_high_resolution_scheme_' in path:
            inner_path = path

    if not outer_path:
        print("TEST SKIPPED: standard_data_create_high_resolution_scheme.pkl not found.")
        sys.exit(0)

    print(f"Loading Outer Data: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # Reconcile types in outer_data
    # This converts __main__.Scheme instances in args/kwargs/output to agent...Scheme
    outer_data['args'] = reconcile_scheme_objects(outer_data['args'])
    outer_data['kwargs'] = reconcile_scheme_objects(outer_data['kwargs'])
    outer_data['output'] = reconcile_scheme_objects(outer_data['output'])

    # Determine Scenario
    if inner_path:
        # Scenario B: Closure / Object creation pattern
        print("Scenario B detected: creating operator then executing inner.")
        try:
            # 1. Create Operator (Agent)
            operator = create_high_resolution_scheme(*outer_data['args'], **outer_data['kwargs'])
            
            # 2. Load Inner Data
            print(f"Loading Inner Data: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            # Reconcile types in inner_data
            inner_data['args'] = reconcile_scheme_objects(inner_data['args'])
            inner_data['kwargs'] = reconcile_scheme_objects(inner_data['kwargs'])
            inner_data['output'] = reconcile_scheme_objects(inner_data['output'])

            # 3. Execute Operator
            actual_result = operator(*inner_data['args'], **inner_data['kwargs'])
            expected_result = inner_data['output']

        except Exception as e:
            print(f"Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: direct function execution.")
        try:
            actual_result = create_high_resolution_scheme(*outer_data['args'], **outer_data['kwargs'])
            expected_result = outer_data['output']
        except Exception as e:
            print(f"Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Verification
    print("Verifying results...")
    
    # Custom check wrapper to handle Scheme objects specifically if recursive_check fails slightly
    # or to ensure attributes are compared if they are custom objects.
    # However, since we reconciled the types earlier, expected_result and actual_result 
    # should now both be instances of agent_create_high_resolution_scheme.Scheme.
    # We rely on recursive_check to compare their __dict__ if they are objects.
    
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            # Fallback: if strict type check failed inside recursive_check despite our reconciliation
            # (e.g. some deep internal difference), we try to compare __dict__ manually for Scheme objects.
            if isinstance(expected_result, Scheme) and isinstance(actual_result, Scheme):
                print("Direct object comparison failed. Attempting attribute comparison...")
                dict_passed, dict_msg = recursive_check(expected_result.__dict__, actual_result.__dict__)
                if dict_passed:
                    print("TEST PASSED (via attribute comparison)")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: Attribute mismatch: {dict_msg}")
                    sys.exit(1)
            
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()