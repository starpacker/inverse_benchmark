import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import target function
try:
    from agent_get_tgc import get_tgc
except ImportError:
    # Fallback/Mock if the file structure is different during testing setup, 
    # but normally this file is expected to exist next to the test.
    print("Error: Could not import 'get_tgc' from 'agent_get_tgc.py'.")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def run_test():
    # Data paths provided in the prompt
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_get_tgc.pkl']
    
    # 1. Logic to distinguish between Simple Function vs Factory Pattern
    # We look for the "outer" data file (standard_data_get_tgc.pkl)
    # and potentially an "inner" data file (standard_data_parent_function_get_tgc_*.pkl).
    
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        filename = os.path.basename(path)
        if filename == 'standard_data_get_tgc.pkl':
            outer_data_path = path
        elif 'standard_data_parent_function_get_tgc' in filename:
            inner_data_path = path

    if not outer_data_path:
        print("Test Skipped: Primary data file 'standard_data_get_tgc.pkl' not found in provided paths.")
        sys.exit(0)

    if not os.path.exists(outer_data_path):
        print(f"Test Skipped: File does not exist at path: {outer_data_path}")
        sys.exit(0)

    # 2. Load Outer Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_data_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    # For Scenario A, the expected output is here. For Scenario B, this might be a function/closure.
    outer_expected = outer_data.get('output', None)

    # 3. Execution Phase 1: Run the Target Function
    try:
        print("Executing get_tgc with outer arguments...")
        phase1_result = get_tgc(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed during call to get_tgc: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine Strategy based on Inner Data presence and Result Type
    # If phase1_result is callable and inner_data_path exists, it's a Closure/Factory (Scenario B).
    # Otherwise, it's a Simple Function (Scenario A).
    
    final_result = phase1_result
    expected_result = outer_expected
    
    if inner_data_path and callable(phase1_result):
        # Scenario B: Factory Pattern
        print(f"Detected Factory Pattern. Loading inner data from {inner_data_path}...")
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output', None)
            
            print("Executing inner operator (closure)...")
            final_result = phase1_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Execution failed during inner operator call: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        # The prompt code shows get_tgc returns a calculated value (tgc_gain), not a function.
        # So we verify phase1_result directly against outer_expected.
        pass

    # 5. Verification
    try:
        print("Verifying results...")
        passed, msg = recursive_check(expected_result, final_result)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()