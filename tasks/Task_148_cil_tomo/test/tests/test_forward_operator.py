import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/cil_tomo_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"WARNING: Data file not found: {path}")
            continue
        
        basename = os.path.basename(path)
        # Check if it's an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Outer data file (standard_data_forward_operator.pkl) not found.")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute forward_operator with outer data
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator with outer data.")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner data file(s).")
        
        # The result should be callable (the operator/closure)
        if not callable(result):
            print("ERROR: Expected forward_operator to return a callable (operator), but it did not.")
            print(f"Result type: {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        print("forward_operator returned a callable operator.")
        
        # Process each inner path
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed the operator with inner data.")
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(inner_output, actual_result)
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}:")
                    print(msg)
                    all_passed = False
                else:
                    print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)
    
    else:
        # Scenario A: Simple function - compare outer output directly
        print("Detected Scenario A: Simple function pattern.")
        
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print("VERIFICATION FAILED:")
                print(msg)
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