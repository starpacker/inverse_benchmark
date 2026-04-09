import sys
import os
import dill
import numpy as np
import torch
import traceback
from verification_utils import recursive_check

# Ensure the current directory is in the path to import the agent code
sys.path.append(os.getcwd())

try:
    from agent_psf_width import psf_width
except ImportError:
    print("Error: Could not import 'psf_width' from 'agent_psf_width.py'.")
    sys.exit(1)

# Helper to ensure custom classes (like simSettings) can be loaded even if context is tricky
try:
    import brighteyes_ism.simulation.PSF_sim as psf_sim
except ImportError:
    # If the library isn't installed, dill might still fail, 
    # but we proceed hoping the environment is set up correctly.
    print("Warning: 'brighteyes_ism' module not found. Dill loading might fail if it relies on class definitions.")
    pass

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_psf_width.pkl']
    
    outer_data_path = None
    inner_data_path = None

    # Logic to distinguish between simple execution and factory pattern
    for path in data_paths:
        if 'standard_data_psf_width.pkl' in path:
            outer_data_path = path
        elif 'standard_data_parent_function_psf_width_' in path:
            inner_data_path = path

    if not outer_data_path or not os.path.exists(outer_data_path):
        print(f"Error: Main data file not found at {outer_data_path}")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    print(f"Loaded input data from {outer_data_path}")
    print(f"Function: {outer_data.get('func_name')}")

    # 3. Execution Phase 1: Run the main function
    try:
        print("Executing psf_width...")
        result_phase_1 = psf_width(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error during execution of psf_width: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification Logic (Factory vs Simple)
    if inner_data_path and os.path.exists(inner_data_path):
        # Scenario B: The result is a callable (operator)
        print("Detected Inner Data file. Treating result as a factory/operator.")
        
        if not callable(result_phase_1):
            print(f"Error: Expected a callable result from Phase 1, but got {type(result_phase_1)}")
            sys.exit(1)

        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner pickle file: {e}")
            sys.exit(1)

        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_final_output = inner_data.get('output', None)

        try:
            print("Executing inner operator...")
            actual_final_result = result_phase_1(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error during execution of inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        passed, msg = recursive_check(expected_final_output, actual_final_result)

    else:
        # Scenario A: Simple function execution
        print("No Inner Data file found. Comparing Phase 1 result directly.")
        actual_final_result = result_phase_1
        expected_final_output = expected_outer_output
        
        passed, msg = recursive_check(expected_final_output, actual_final_result)

    # 5. Final Report
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected_final_output}")
        print(f"Actual:   {actual_final_result}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()