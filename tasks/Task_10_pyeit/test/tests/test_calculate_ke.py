import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_calculate_ke import calculate_ke

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for calculate_ke."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_calculate_ke.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_calculate_ke.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_calculate_ke.pkl in data_paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on presence of inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Execute outer function to get the operator/closure
        try:
            print("Executing calculate_ke to obtain operator...")
            agent_operator = calculate_ke(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute calculate_ke: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute operator
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Execute the function directly
        try:
            print("Executing calculate_ke...")
            result = calculate_ke(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute calculate_ke: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Expected output from outer data
        expected = outer_output
        
        # Comparison
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()