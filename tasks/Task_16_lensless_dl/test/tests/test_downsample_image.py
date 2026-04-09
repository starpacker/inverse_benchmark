import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path to import the target module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_downsample_image import downsample_image
from verification_utils import recursive_check


def main():
    """Main test function for downsample_image."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_downsample_image.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_downsample_image.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if not inner_paths and outer_path:
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Args: {len(args)} arguments")
            print(f"Kwargs: {list(kwargs.keys())}")
            
        except Exception as e:
            print(f"FAILED: Could not load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            result = downsample_image(*args, **kwargs)
            print("Function executed successfully.")
            
        except Exception as e:
            print(f"FAILED: Function execution error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"FAILED: Verification failed - {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"FAILED: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (inner paths exist)
    elif inner_paths and outer_path:
        try:
            # Phase 1: Load outer data and reconstruct operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Outer Args: {len(outer_args)} arguments")
            print(f"Outer Kwargs: {list(outer_kwargs.keys())}")
            
        except Exception as e:
            print(f"FAILED: Could not load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Create the operator/closure
            agent_operator = downsample_image(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"FAILED: downsample_image did not return a callable, got {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully.")
            
        except Exception as e:
            print(f"FAILED: Operator creation error: {e}")
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
                print(f"Inner Args: {len(inner_args)} arguments")
                print(f"Inner Kwargs: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"FAILED: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Execute the operator with inner arguments
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"FAILED: Operator execution error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Compare results
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"FAILED: Verification failed for {os.path.basename(inner_path)} - {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"FAILED: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print(f"FAILED: No valid data paths found. Outer: {outer_path}, Inner: {inner_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()