import sys
import os
import dill
import numpy as np
import traceback

# Handle conditional torch import to fix ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_CT import CT
from verification_utils import recursive_check

def run_test():
    # Paths provided in the prompt context
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_CT.pkl']

    # 1. DATA FILE ANALYSIS
    outer_data_path = None
    inner_data_paths = []

    for p in data_paths:
        if p.endswith('standard_data_CT.pkl'):
            outer_data_path = p
        elif 'parent_function_CT' in p:
            inner_data_paths.append(p)

    if not outer_data_path:
        print("Error: Could not find standard_data_CT.pkl in provided paths.")
        sys.exit(1)

    # 2. LOAD DATA
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"Error loading data from {outer_data_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. EXECUTION PHASE
    try:
        print("Executing CT function...")
        # Scenario A: Direct execution of CT
        actual_result = CT(*outer_args, **outer_kwargs)
        
        # Check if Scenario B (Closure) is applicable
        # If the result is callable and we have inner data, treat as factory.
        if callable(actual_result) and inner_data_paths:
            print("Detected Closure/Factory pattern. Testing inner function execution...")
            # We iterate through inner data files if any (though none provided in this specific prompt context)
            for inner_path in inner_data_paths:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output', None)
                
                actual_inner_result = actual_result(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_inner_output, actual_inner_result)
                if not passed:
                    print(f"TEST FAILED on inner data {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            
            # If we processed inner files successfully
            print("TEST PASSED")
            sys.exit(0)

        # Scenario A: Simple Function (Target scenario for the provided CT code)
        else:
            expected_output = expected_outer_output
            
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFICATION (For Scenario A)
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()