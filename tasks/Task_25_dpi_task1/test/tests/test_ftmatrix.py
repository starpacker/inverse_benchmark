import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_ftmatrix import ftmatrix

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for ftmatrix."""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task1_sandbox/run_code/std_data/standard_data_ftmatrix.pkl']
    
    # Analyze data paths to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_ftmatrix.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer data exists
    # Scenario B: Factory/Closure - both outer and inner data exist
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_ftmatrix.pkl")
        sys.exit(1)
    
    print(f"Found outer data: {outer_path}")
    print(f"Found inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Successfully loaded outer data")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args: {len(outer_args)} positional arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = ftmatrix(*outer_args, **outer_kwargs)
        print("Successfully executed ftmatrix")
    except Exception as e:
        print(f"ERROR executing ftmatrix: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Successfully loaded inner data")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Inner args: {len(inner_args)} positional arguments")
            print(f"Inner kwargs: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed operator")
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner data")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        
        # Verify result directly against expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            print("Verification passed")
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()