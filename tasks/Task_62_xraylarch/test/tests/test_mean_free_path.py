import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_mean_free_path import mean_free_path
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/xraylarch_sandbox_sandbox/run_code/std_data/standard_data_mean_free_path.pkl']

def main():
    try:
        # Identify outer and inner paths
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_mean_free_path.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_mean_free_path.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the target function
        print("Executing mean_free_path with outer args/kwargs...")
        agent_result = mean_free_path(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine if this is Scenario A or Scenario B
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
            
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator but got {type(agent_result)}")
                sys.exit(1)
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                print("Executing operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
        else:
            # Scenario A: Simple function
            print("Scenario A detected: Simple function test")
            result = agent_result
            expected = outer_output
            
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Exception occurred during test execution")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()