import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_fft2c import fft2c
from verification_utils import recursive_check

def main():
    """Main test function for fft2c."""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_fft2c.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_fft2c.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_fft2c.pkl in data paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        # Execute the function with outer args
        agent_result = fft2c(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute fft2c with outer args")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable operator from fft2c, but got non-callable")
            print(f"Result type: {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner args from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        # The result from Phase 1 IS the final result
        expected = outer_data.get('output')
        result = agent_result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()