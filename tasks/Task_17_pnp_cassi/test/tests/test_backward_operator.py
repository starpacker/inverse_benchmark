import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_backward_operator import backward_operator

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for backward_operator."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pnp_cassi_sandbox/run_code/std_data/standard_data_backward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_backward_operator.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        try:
            # Load the outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Number of args: {len(args)}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {outer_path}")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            result = backward_operator(*args, **kwargs)
            print("Function executed successfully.")
            
        except Exception as e:
            print(f"FAILED: Error executing backward_operator")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"FAILED: Result mismatch")
                print(f"Message: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"FAILED: Error during result comparison")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (has inner paths)
    elif outer_path and inner_paths:
        try:
            # Phase 1: Load outer data and reconstruct operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Number of outer args: {len(outer_args)}")
            print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
            
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {outer_path}")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Create the operator/closure
            agent_operator = backward_operator(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"FAILED: backward_operator did not return a callable, got {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully.")
            
        except Exception as e:
            print(f"FAILED: Error creating operator from backward_operator")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Loaded inner data from: {inner_path}")
                print(f"Number of inner args: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"FAILED: Could not load inner data file: {inner_path}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Execute the operator with inner args
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully with inner data.")
                
            except Exception as e:
                print(f"FAILED: Error executing operator with inner data")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Compare results
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"FAILED: Result mismatch for {os.path.basename(inner_path)}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"FAILED: Error during result comparison for {os.path.basename(inner_path)}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print(f"FAILED: No valid outer data path found in: {data_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()