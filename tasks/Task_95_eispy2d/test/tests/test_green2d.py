import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_green2d import green2d
from verification_utils import recursive_check

def main():
    """Main test function for green2d."""
    
    # Data paths provided
    data_paths = ['/data/yjh/eispy2d_sandbox_sandbox/run_code/std_data/standard_data_green2d.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_green2d.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_green2d.pkl in data_paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct/call the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {len(outer_args)} arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function with outer args
    try:
        agent_result = green2d(*outer_args, **outer_kwargs)
        print("Successfully executed green2d with outer arguments")
    except Exception as e:
        print(f"ERROR: Failed to execute green2d: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from green2d, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {len(inner_args)} arguments")
            print(f"Inner kwargs: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed operator with inner arguments")
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
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function test")
        
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

if __name__ == "__main__":
    main()