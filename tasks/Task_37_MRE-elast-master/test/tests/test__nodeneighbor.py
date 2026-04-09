import sys
import os
import dill
import numpy as np
import traceback

# Safe import for torch, as the environment might not have it
# and the function under test appears to be pure numpy.
try:
    import torch
except ImportError:
    torch = None

from agent__nodeneighbor import _nodeneighbor
from verification_utils import recursive_check

def test_nodeneighbor():
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data__nodeneighbor.pkl']
    
    # Identify Outer and Inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if "standard_data__nodeneighbor.pkl" in path:
            outer_path = path
        elif "parent_function" in path:
            inner_paths.append(path)
            
    if not outer_path:
        print("Error: standard_data__nodeneighbor.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        # If the pickle contains torch objects and torch is missing, this might fail.
        # But we've handled the explicit import error.
        traceback.print_exc()
        sys.exit(1)

    # 1. Execute the Primary Function
    print("Executing _nodeneighbor with loaded arguments...")
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Execute
        result = _nodeneighbor(*args, **kwargs)
        
    except Exception as e:
        print(f"Execution of _nodeneighbor failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Determine Strategy (Simple vs Factory)
    # Check if the result is callable (Factory pattern) AND if we have inner data files
    if callable(result) and inner_paths:
        print("Detected Factory/Closure pattern. Proceeding to test inner function execution.")
        
        agent_operator = result
        all_passed = True
        
        for inner_path in inner_paths:
            print(f"  Testing against inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner = inner_data.get('output')
                
                # Execute inner
                actual_inner = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner, actual_inner)
                if not passed:
                    print(f"    FAILED: {msg}")
                    all_passed = False
                else:
                    print("    PASSED")
                    
            except Exception as e:
                print(f"    Error processing inner file {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("TEST PASSED (Factory Pattern)")
            sys.exit(0)
        else:
            print("TEST FAILED (Factory Pattern)")
            sys.exit(1)

    else:
        # Scenario A: Simple Function
        print("Detected Simple Function pattern. verifying output directly.")
        expected_output = outer_data.get('output')
        
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    test_nodeneighbor()