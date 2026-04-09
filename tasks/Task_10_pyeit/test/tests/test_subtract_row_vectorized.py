import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_subtract_row_vectorized import subtract_row_vectorized

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for subtract_row_vectorized."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_subtract_row_vectorized.pkl']
    
    # Determine test scenario by analyzing file paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_subtract_row_vectorized.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if not inner_paths:
        print("Detected Scenario A: Simple Function Test")
        
        if outer_path is None:
            print("ERROR: No standard data file found.")
            sys.exit(1)
        
        # Load outer data
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"Successfully loaded outer data from: {outer_path}")
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        try:
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            print(f"Outer args count: {len(outer_args)}")
            print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        except Exception as e:
            print(f"ERROR: Failed to extract data from outer_data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Execute the function
        try:
            result = subtract_row_vectorized(*outer_args, **outer_kwargs)
            print("Successfully executed subtract_row_vectorized")
        except Exception as e:
            print(f"ERROR: Failed to execute subtract_row_vectorized: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed during result comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure Pattern
    else:
        print("Detected Scenario B: Factory/Closure Pattern Test")
        
        if outer_path is None:
            print("ERROR: No outer data file found for factory pattern.")
            sys.exit(1)
        
        # Phase 1: Load outer data and create operator
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"Successfully loaded outer data from: {outer_path}")
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
        except Exception as e:
            print(f"ERROR: Failed to extract outer args/kwargs: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Create the operator/closure
        try:
            agent_operator = subtract_row_vectorized(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
            except Exception as e:
                print(f"ERROR: Failed to extract inner args/kwargs: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected_output, result)
                if passed:
                    print(f"Inner test PASSED for: {inner_path}")
                else:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Failed during result comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()