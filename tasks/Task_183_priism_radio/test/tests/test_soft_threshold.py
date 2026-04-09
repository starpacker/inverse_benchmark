import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_soft_threshold import soft_threshold
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_soft_threshold.pkl']

def main():
    try:
        # Classify data files
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_soft_threshold.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_soft_threshold.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and call the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the target function
        print("Executing soft_threshold with outer data...")
        try:
            result = soft_threshold(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR executing soft_threshold: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Determine if this is a factory pattern or simple function
        if inner_paths and callable(result):
            # Scenario B: Factory/Closure pattern
            print(f"Detected factory pattern. Result is callable: {type(result)}")
            agent_operator = result
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Executing agent_operator with inner data...")
                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR executing agent_operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verification
                print("Verifying results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
            
            print("TEST PASSED")
            sys.exit(0)
        else:
            # Scenario A: Simple function
            print("Detected simple function pattern.")
            expected = outer_output
            
            # Verification
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()