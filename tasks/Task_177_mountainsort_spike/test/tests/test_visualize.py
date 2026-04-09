import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_visualize import visualize
from verification_utils import recursive_check

def main():
    """Main test function for visualize."""
    
    # Define data paths
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_visualize.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing visualize function...")
        result = visualize(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing visualize: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern)
    if callable(result) and inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected callable result with inner data - using factory pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Executing agent operator with inner args...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED for inner path {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with expected output
        print("Using simple function pattern - comparing result with expected output")
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()