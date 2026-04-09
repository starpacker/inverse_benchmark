import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_prepare_4d_array import prepare_4d_array

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for prepare_4d_array."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_prepare_4d_array.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_prepare_4d_array.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_prepare_4d_array.pkl")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file not found: {outer_path}")
        sys.exit(1)
    
    try:
        # Load the outer data
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract args and kwargs from outer data
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Loaded outer data from: {outer_path}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Scenario determination
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        try:
            # Phase 1: Create the operator
            agent_operator = prepare_4d_array(*outer_args, **outer_kwargs)
            
            # Verify it's callable
            if not callable(agent_operator):
                print("ERROR: prepare_4d_array did not return a callable operator")
                sys.exit(1)
            
            print("Phase 1: Successfully created operator")
            
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"ERROR: Inner data file not found: {inner_path}")
                sys.exit(1)
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Loaded inner data from: {inner_path}")
                
                # Execute the operator
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify the result
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        try:
            # Execute the function directly
            result = prepare_4d_array(*outer_args, **outer_kwargs)
            
            print("Successfully executed prepare_4d_array")
            
            # Verify the result
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()