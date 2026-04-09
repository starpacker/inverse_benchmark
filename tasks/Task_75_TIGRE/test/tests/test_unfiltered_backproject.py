import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/TIGRE_sandbox_sandbox/run_code')

from agent_unfiltered_backproject import unfiltered_backproject
from verification_utils import recursive_check

def main():
    """Test unfiltered_backproject function."""
    
    # Data paths provided
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_unfiltered_backproject.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_unfiltered_backproject.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_unfiltered_backproject.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("Executing unfiltered_backproject...")
        result = unfiltered_backproject(*outer_args, **outer_kwargs)
        print(f"Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing unfiltered_backproject: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Scenario A: Simple function - no inner paths
    # The result from the function call is the final result
    if len(inner_paths) == 0:
        print("Scenario A: Simple function (no inner data files)")
        
        # Compare result with expected output
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B: Factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print(f"Executing agent operator...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = recursive_check(inner_expected, actual_result)
                
                if passed:
                    print(f"PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()