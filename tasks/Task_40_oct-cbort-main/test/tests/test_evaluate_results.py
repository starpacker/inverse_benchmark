import sys
import os
import dill
import numpy as np
import torch
import traceback
import matplotlib

# Set backend to Agg to prevent GUI errors during testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Provided data paths
data_paths = ['/data/yjh/oct-cbort-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def load_pkl(path):
    """Helper to load dill/pickle files."""
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Failed to load data from {path}: {e}")
        sys.exit(1)

def run_test():
    print("Starting test_evaluate_results.py...")
    
    # 1. Analyze Data Files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results' in path:
            inner_paths.append(path)
            
    if not outer_path:
        print("CRITICAL: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    try:
        # 2. Phase 1: Reconstruct/Execute Primary Operator
        print(f"Loading primary data from {outer_path}...")
        outer_data = load_pkl(outer_path)
        
        # Extract inputs safely
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_result = outer_data.get('output')
        
        print("Executing evaluate_results...")
        # Execute the function
        primary_result = evaluate_results(*outer_args, **outer_kwargs)
        
        # 3. Phase 2: Strategy Determination (Simple vs Factory)
        
        # Check if the result is a closure that needs further execution
        # Scenario B condition: Result is callable AND we have inner data files to run against it
        is_closure_pattern = callable(primary_result) and len(inner_paths) > 0
        
        if is_closure_pattern:
            print(f"Detected Factory/Closure pattern. Found {len(inner_paths)} inner execution files.")
            agent_operator = primary_result
            
            # Iterate through all inner data files to verify the closure
            for i_path in inner_paths:
                print(f"  Verifying against inner data: {os.path.basename(i_path)}")
                inner_data = load_pkl(i_path)
                
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                expected_inner_result = inner_data.get('output')
                
                # Execute closure
                actual_inner_result = agent_operator(*i_args, **i_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner_result, actual_inner_result)
                if not passed:
                    print(f"  FAILED inner verification for {os.path.basename(i_path)}: {msg}")
                    sys.exit(1)
                print(f"  Passed inner verification for {os.path.basename(i_path)}")
                
        else:
            # Scenario A: Simple Function (The result is the final value)
            print("Detected Simple Function pattern.")
            
            passed, msg = recursive_check(expected_outer_result, primary_result)
            if not passed:
                print(f"FAILED verification: {msg}")
                print(f"Expected: {expected_outer_result}")
                print(f"Actual:   {primary_result}")
                sys.exit(1)
            else:
                print("Verification successful.")

        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"Execution failed with unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()