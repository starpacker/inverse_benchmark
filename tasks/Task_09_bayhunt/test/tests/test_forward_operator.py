import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/home/yjh/BayHunter_standalone/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl in data_paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
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
    expected_output = outer_data.get('output', None)
    
    print(f"Loaded outer data with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
    
    # Execute forward_operator
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected Scenario B: Factory/Closure Pattern with {len(inner_paths)} inner data file(s)")
        
        # The result should be a callable (operator/closure)
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"Processing inner data: {inner_path}")
            
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
            inner_expected = inner_data.get('output', None)
            
            print(f"Inner data has {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed operator with inner arguments")
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner arguments")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner data verification passed for: {inner_path}")
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Compare result directly with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()