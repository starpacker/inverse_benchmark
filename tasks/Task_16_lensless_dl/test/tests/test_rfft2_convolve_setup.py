import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_rfft2_convolve_setup import rfft2_convolve_setup

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for rfft2_convolve_setup."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_rfft2_convolve_setup.pkl']
    
    # Classify data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"FAILED: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (parent_function pattern)
        if 'parent_function' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact match for the function)
        elif basename == 'standard_data_rfft2_convolve_setup.pkl':
            outer_path = path
    
    if outer_path is None:
        print("FAILED: Could not find outer data file (standard_data_rfft2_convolve_setup.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct the operator/result
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAILED: Could not load outer data file: {outer_path}")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    # Execute the target function
    try:
        agent_result = rfft2_convolve_setup(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAILED: Error executing rfft2_convolve_setup with outer data")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine test scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable, and we need to execute it with inner data
        
        if not callable(agent_result):
            print("FAILED: Expected rfft2_convolve_setup to return a callable operator, but it did not.")
            print(f"Returned type: {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED: Could not load inner data file: {inner_path}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            # Execute the operator with inner args
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Error executing operator with inner data from {inner_path}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected_output, actual_result)
            except Exception as e:
                print(f"FAILED: Error during result comparison for {inner_path}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"FAILED: Result mismatch for inner data {inner_path}")
                print(f"Details: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        # The result from rfft2_convolve_setup is the final result to compare
        
        expected_output = outer_output
        actual_result = agent_result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"FAILED: Error during result comparison")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: Result mismatch")
            print(f"Details: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()