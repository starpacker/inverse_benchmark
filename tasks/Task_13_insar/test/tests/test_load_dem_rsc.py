import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_dem_rsc import load_dem_rsc

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for load_dem_rsc"""
    
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_load_dem_rsc.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_dem_rsc.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_load_dem_rsc.pkl in data_paths")
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
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing load_dem_rsc with outer args/kwargs...")
        agent_result = load_dem_rsc(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute load_dem_rsc: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator but got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Extract inner args and kwargs
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
            except Exception as e:
                print(f"ERROR: Failed to extract inner args/kwargs: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully. Result type: {type(result)}")
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
        print("Scenario A detected: Simple function test")
        
        # The result from Phase 1 is the final result
        result = agent_result
        expected = outer_data.get('output')
        
        # Verify results
        try:
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
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