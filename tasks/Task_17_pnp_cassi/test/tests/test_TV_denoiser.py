import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_TV_denoiser import TV_denoiser

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for TV_denoiser."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pnp_cassi_sandbox/run_code/std_data/standard_data_TV_denoiser.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_TV_denoiser.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find standard_data_TV_denoiser.pkl in data_paths")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Execute the target function
        print("Executing TV_denoiser with outer args/kwargs...")
        result = TV_denoiser(*outer_args, **outer_kwargs)
        print(f"Execution completed. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute TV_denoiser: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or Scenario B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator)
        if not callable(result):
            print(f"WARNING: Result is not callable (type: {type(result)}), treating as Scenario A")
            # Fall back to Scenario A
            expected = outer_output
        else:
            # Process inner data
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args/kwargs...")
                result = result(*inner_args, **inner_kwargs)
                expected = inner_output
                
            except Exception as e:
                print(f"ERROR: Failed to process inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function execution")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Verifying results...")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        
        passed, msg = recursive_check(expected, result)
        
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