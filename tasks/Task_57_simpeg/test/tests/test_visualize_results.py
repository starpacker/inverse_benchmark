import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_visualize_results import visualize_results
from verification_utils import recursive_check

def main():
    """Main test function for visualize_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_visualize_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("Executing visualize_results...")
        result = visualize_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        
    except Exception as e:
        print(f"ERROR executing visualize_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and inner paths exist)
    if inner_paths and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print("Executing agent operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly
        print("Detected simple function pattern")
        
        try:
            passed, msg = recursive_check(expected_output, result)
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