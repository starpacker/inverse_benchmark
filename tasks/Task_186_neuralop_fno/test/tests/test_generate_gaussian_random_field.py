import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_generate_gaussian_random_field import generate_gaussian_random_field

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for generate_gaussian_random_field."""
    
    # Data paths provided
    data_paths = ['/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_generate_gaussian_random_field.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_gaussian_random_field.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_gaussian_random_field.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing generate_gaussian_random_field with outer args/kwargs...")
        agent_result = generate_gaussian_random_field(*outer_args, **outer_kwargs)
        print(f"Function returned type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute generate_gaussian_random_field: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator)
        if not callable(agent_result):
            print("WARNING: Result is not callable, treating as Scenario A instead")
            # Fall through to Scenario A logic
            result = agent_result
            expected = outer_output
        else:
            # Process inner data
            inner_path = inner_paths[0]  # Use first inner path
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                expected = inner_output
                print(f"Operator returned type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call")
        result = agent_result
        expected = outer_output
    
    # Comparison
    try:
        print("Running recursive_check comparison...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()