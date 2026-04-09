import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in python path to import the agent code
sys.path.append(os.getcwd())

from agent_aux_structures_generate import aux_structures_generate
from verification_utils import recursive_check

def run_test():
    # 1. DATA FILE ANALYSIS
    # Provided paths from the prompt
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_aux_structures_generate.pkl']
    
    # We look for the primary data file.
    # Scenario A: Only standard_data_aux_structures_generate.pkl exists.
    # Scenario B: standard_data_aux_structures_generate.pkl AND standard_data_parent_function_aux_structures_generate_*.pkl exist.
    
    outer_data_path = None
    inner_data_paths = []

    for path in data_paths:
        if "standard_data_aux_structures_generate.pkl" in path:
            outer_data_path = path
        elif "standard_data_parent_function_aux_structures_generate_" in path:
            inner_data_paths.append(path)

    if not outer_data_path:
        print("CRITICAL: Primary data file 'standard_data_aux_structures_generate.pkl' not found.")
        sys.exit(1)

    # 2. LOAD DATA
    try:
        print(f"Loading data from {outer_data_path}...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    
    # 3. EXECUTE TARGET FUNCTION
    print("Executing aux_structures_generate...")
    try:
        # Phase 1: Run the main function
        result_object = aux_structures_generate(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFICATION STRATEGY
    # Check if we are in Scenario B (Closure/Factory) or Scenario A (Simple Result)
    
    if inner_data_paths:
        # SCENARIO B: The result_object is expected to be a callable (operator)
        print("Scenario B detected: Testing inner operator execution.")
        
        if not callable(result_object):
            print(f"TEST FAILED: Expected a callable operator, but got {type(result_object)}")
            sys.exit(1)

        for inner_path in inner_data_paths:
            print(f"Testing against inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Skipping inner file {inner_path} due to load error: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')

            try:
                actual_inner_output = result_object(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected_inner_output, actual_inner_output)
            if not passed:
                print(f"TEST FAILED on inner data comparison: {msg}")
                sys.exit(1)
        
        # If loop completes without exit, passed
        print("TEST PASSED")
        sys.exit(0)

    else:
        # SCENARIO A: The result_object is the final result
        print("Scenario A detected: Verifying direct function output.")
        expected_output = outer_data.get('output')
        
        passed, msg = recursive_check(expected_output, result_object)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()