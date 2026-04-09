import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__line_pixel_lengths import _line_pixel_lengths

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for _line_pixel_lengths"""
    
    # Define data paths
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data__line_pixel_lengths.pkl']
    
    # Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is inner data (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__line_pixel_lengths.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer data exists
    if outer_path is not None and len(inner_paths) == 0:
        print("Detected Scenario A: Simple Function")
        
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"Loaded outer data from: {outer_path}")
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Function name: {outer_data.get('func_name')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Number of kwargs: {len(outer_kwargs)}")
        
        try:
            # Execute the function
            result = _line_pixel_lengths(*outer_args, **outer_kwargs)
            print("Function executed successfully")
        except Exception as e:
            print(f"ERROR: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify result
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure Pattern
    elif outer_path is not None and len(inner_paths) > 0:
        print("Detected Scenario B: Factory/Closure Pattern")
        
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"Loaded outer data from: {outer_path}")
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract outer args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        try:
            # Phase 1: Create the operator/closure
            agent_operator = _line_pixel_lengths(*outer_args, **outer_kwargs)
            print("Created agent operator successfully")
            
            # Verify it's callable
            if not callable(agent_operator):
                print("ERROR: Agent operator is not callable")
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
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                # Execute the operator with inner data
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully")
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected_output, result)
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print("ERROR: No valid data files found")
        sys.exit(1)

if __name__ == "__main__":
    main()