import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_p_shrink import p_shrink

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for p_shrink."""
    
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_p_shrink.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_p_shrink.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_p_shrink.pkl in data_paths")
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
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to extract outer data components: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on presence of inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Execute outer function to get operator
        try:
            print("Executing p_shrink to get operator...")
            agent_operator = p_shrink(*outer_args, **outer_kwargs)
            
            # Verify operator is callable
            if not callable(agent_operator):
                print(f"ERROR: Result is not callable. Got type: {type(agent_operator)}")
                sys.exit(1)
            print("Successfully created callable operator")
        except Exception as e:
            print(f"ERROR: Failed to execute p_shrink for operator creation: {e}")
            traceback.print_exc()
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
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR: Failed to extract inner data components: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute operator with inner args
            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                print("Verifying results...")
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Execute function directly
        try:
            print("Executing p_shrink...")
            result = p_shrink(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute p_shrink: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify results
        try:
            print("Verifying results...")
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()