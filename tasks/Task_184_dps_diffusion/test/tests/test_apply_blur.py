import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_apply_blur import apply_blur

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for apply_blur."""
    
    # Data paths provided
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_apply_blur.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_apply_blur.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_apply_blur.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Load outer data
    try:
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
    outer_output = outer_data.get('output')
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Phase 1: Reconstruct Operator
        try:
            agent_operator = apply_blur(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure")
        except Exception as e:
            print(f"ERROR: Failed to create operator in Phase 1: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print("ERROR: Created operator is not callable")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
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
            expected = inner_data.get('output')
            
            # Execute operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute operator in Phase 2: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Execute the function directly
        try:
            result = apply_blur(*outer_args, **outer_kwargs)
            print("Successfully executed apply_blur")
        except Exception as e:
            print(f"ERROR: Failed to execute apply_blur: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()