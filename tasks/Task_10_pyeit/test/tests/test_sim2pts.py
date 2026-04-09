import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_sim2pts import sim2pts

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for sim2pts"""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_sim2pts.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_sim2pts.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths found)
    # This is the case based on the provided data_paths
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_sim2pts.pkl in data_paths")
        sys.exit(1)
    
    print(f"Found outer data path: {outer_path}")
    print(f"Found {len(inner_paths)} inner data path(s)")
    
    # Phase 1: Load outer data and execute function
    try:
        print("Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Outer data loaded successfully")
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
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing sim2pts function...")
        result = sim2pts(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute sim2pts: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Verify the result is callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Inner data loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("Executing agent operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Agent operator executed successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple Function")
        # result is already set from the function call above
        # expected_output is already set from outer_data
    
    # Phase 2: Verification
    try:
        print("Verifying results...")
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected_output)}")
        
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        if isinstance(expected_output, np.ndarray):
            print(f"Expected shape: {expected_output.shape}, dtype: {expected_output.dtype}")
        
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()