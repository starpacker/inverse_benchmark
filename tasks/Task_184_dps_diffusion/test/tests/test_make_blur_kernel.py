import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_make_blur_kernel import make_blur_kernel

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for make_blur_kernel."""
    
    # Define data paths
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_make_blur_kernel.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if it's an inner data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        # Check if it's the outer data file (exact match pattern)
        elif filename == 'standard_data_make_blur_kernel.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_make_blur_kernel.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function to get the operator/result
    try:
        agent_result = make_blur_kernel(*outer_args, **outer_kwargs)
        print(f"Successfully executed make_blur_kernel, result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute make_blur_kernel: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Verify that the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator but got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data: {inner_data.get('func_name', 'unknown')}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Extract inner args and kwargs
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner data
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed operator, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print("TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print("Verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()