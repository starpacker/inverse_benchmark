import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_chambolle_tv_prox import chambolle_tv_prox

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for chambolle_tv_prox."""
    
    # Data paths provided
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_chambolle_tv_prox.pkl']
    
    # Classify data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_chambolle_tv_prox.pkl':
            outer_path = path
    
    # Validate outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_chambolle_tv_prox.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct/execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function with outer args
    try:
        print("Executing chambolle_tv_prox with outer arguments...")
        agent_result = chambolle_tv_prox(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute chambolle_tv_prox: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    # Filter inner paths to only existing files
    existing_inner_paths = [p for p in inner_paths if os.path.exists(p)]
    
    if existing_inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected Scenario B: Factory/Closure Pattern with {len(existing_inner_paths)} inner data file(s)")
        
        # Verify the result is callable (an operator)
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator from chambolle_tv_prox, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner data file
        all_passed = True
        for inner_path in existing_inner_paths:
            try:
                print(f"\nProcessing inner data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"  Inner args count: {len(inner_args)}")
                print(f"  Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}")
                    print(f"Failure message: {msg}")
                    all_passed = False
                else:
                    print(f"  Verification PASSED for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed processing inner data {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if not all_passed:
            print("\nTEST FAILED: One or more inner data verifications failed.")
            sys.exit(1)
        else:
            print("\nTEST PASSED: All inner data verifications succeeded.")
            sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function (no inner data files)")
        
        # The result from Phase 1 is the final result
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"VERIFICATION FAILED")
                print(f"Failure message: {msg}")
                
                # Print additional debug info
                if hasattr(expected, 'shape'):
                    print(f"Expected shape: {expected.shape}")
                if hasattr(result, 'shape'):
                    print(f"Actual shape: {result.shape}")
                if hasattr(expected, 'dtype'):
                    print(f"Expected dtype: {expected.dtype}")
                if hasattr(result, 'dtype'):
                    print(f"Actual dtype: {result.dtype}")
                    
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()