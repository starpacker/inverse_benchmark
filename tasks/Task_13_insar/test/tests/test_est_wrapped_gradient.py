import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_est_wrapped_gradient import est_wrapped_gradient

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for est_wrapped_gradient"""
    
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_est_wrapped_gradient.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_est_wrapped_gradient.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_est_wrapped_gradient.pkl in data_paths")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract args and kwargs from outer data
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        try:
            # Create the operator/closure
            print("Creating operator from est_wrapped_gradient...")
            agent_operator = est_wrapped_gradient(*outer_args, **outer_kwargs)
            
            # Verify it's callable
            if not callable(agent_operator):
                print(f"ERROR: Result is not callable, got type: {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"ERROR: Inner data file does not exist: {inner_path}")
                sys.exit(1)
            
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                
                print(f"Inner test passed for: {inner_path}")
                
            except Exception as e:
                print(f"ERROR: Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        try:
            # Execute the function directly
            print("Executing est_wrapped_gradient...")
            result = est_wrapped_gradient(*outer_args, **outer_kwargs)
            
            # Get expected output from outer data
            expected = outer_data.get('output')
            
            if expected is None:
                print("WARNING: No 'output' key found in outer data")
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"ERROR: Failed during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()