import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_plot_results import plot_results
from verification_utils import recursive_check

def main():
    """Main test function for plot_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_plot_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("Executing plot_results with outer data...")
    try:
        result = plot_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern - result should be callable
        print(f"Factory pattern detected. Found {len(inner_paths)} inner data file(s).")
        
        if not callable(result):
            print(f"ERROR: Expected callable result from factory, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process inner data
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            print("Executing agent operator with inner data...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            print("Verifying results...")
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed for {os.path.basename(inner_path)}")
    else:
        # Scenario A: Simple function - compare result directly
        print("Simple function pattern detected.")
        expected = outer_output
        
        # Verify results
        print("Verifying results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()