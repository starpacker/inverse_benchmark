import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_create_test_image import create_test_image
from verification_utils import recursive_check

def main():
    """Main test function for create_test_image."""
    
    # Data paths provided
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_create_test_image.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_create_test_image.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_test_image.pkl)")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function with outer args
    try:
        result = create_test_image(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute create_test_image: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B - factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable (an operator)
        if not callable(result):
            print("ERROR: Expected callable operator from create_test_image, but got non-callable result")
            sys.exit(1)
        
        agent_operator = result
        print("Phase 1 complete: Got callable operator from create_test_image")
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during result comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed for: {inner_path}")
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - compare result directly with outer output
        expected = outer_output
        
        print("Scenario A: Simple function test")
        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during result comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()