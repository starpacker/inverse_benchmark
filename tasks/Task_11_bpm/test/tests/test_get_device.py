import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_get_device import get_device

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for get_device."""
    
    # Data paths provided
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_get_device.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_get_device.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find standard_data_get_device.pkl in data_paths")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract args and kwargs from outer data
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Execute the target function
        print("Executing get_device()...")
        result = get_device(*outer_args, **outer_kwargs)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute get_device: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Verify the result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
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
                expected = inner_data.get('output', None)
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator with inner args
                print("Executing agent_operator with inner args...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner data processing: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        try:
            # Compare result directly with expected output
            print("Comparing results...")
            print(f"Expected: {expected_output}")
            print(f"Actual: {result}")
            
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()