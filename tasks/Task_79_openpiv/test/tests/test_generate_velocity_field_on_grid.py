import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_velocity_field_on_grid import generate_velocity_field_on_grid

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for generate_velocity_field_on_grid"""
    
    # Data paths provided
    data_paths = ['/data/yjh/openpiv_sandbox_sandbox/run_code/std_data/standard_data_generate_velocity_field_on_grid.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            else:
                outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_velocity_field_on_grid.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if there are inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Factory/Closure Pattern (inner data files found)")
        
        try:
            # Phase 1: Create the operator/closure
            agent_operator = generate_velocity_field_on_grid(*outer_args, **outer_kwargs)
            print("Successfully created agent operator")
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Simple Function Pattern (no inner data files)")
        
        try:
            # Execute the function directly
            result = generate_velocity_field_on_grid(*outer_args, **outer_kwargs)
            print("Successfully executed generate_velocity_field_on_grid")
        except Exception as e:
            print(f"ERROR: Failed to execute generate_velocity_field_on_grid: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("Verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()