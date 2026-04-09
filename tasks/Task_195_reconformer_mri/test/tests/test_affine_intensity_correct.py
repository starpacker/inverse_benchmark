import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_affine_intensity_correct import affine_intensity_correct

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for affine_intensity_correct."""
    
    # Data paths provided
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_affine_intensity_correct.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_affine_intensity_correct.pkl':
            outer_path = path
    
    # Validate outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_affine_intensity_correct.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing affine_intensity_correct with outer args/kwargs...")
        result = affine_intensity_correct(*outer_args, **outer_kwargs)
        print("Function execution completed.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute affine_intensity_correct: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner paths (Scenario B: Factory/Closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable
        print(f"Found {len(inner_paths)} inner data file(s). Checking for closure pattern...")
        
        if not callable(result):
            print("WARNING: Result is not callable but inner paths exist. Treating as simple function.")
            # Fall through to simple comparison
            expected = outer_output
        else:
            # Process inner paths
            agent_operator = result
            
            for inner_path in inner_paths:
                if not os.path.exists(inner_path):
                    print(f"WARNING: Inner data file does not exist: {inner_path}")
                    continue
                
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output')
                    
                    print(f"Inner args count: {len(inner_args)}")
                    print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                    # Execute the operator with inner args
                    print("Executing agent_operator with inner args/kwargs...")
                    result = agent_operator(*inner_args, **inner_kwargs)
                    expected = inner_output
                    
                except Exception as e:
                    print(f"ERROR: Failed to process inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        # Scenario A: Simple function - compare directly with outer output
        print("No inner data files found. Using simple function comparison.")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Running verification...")
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


if __name__ == '__main__':
    main()