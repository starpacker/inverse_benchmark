import sys
import os
import dill
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/nmrglue_sandbox_sandbox/run_code')

import numpy as np

from agent_soft_threshold import soft_threshold
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/nmrglue_sandbox_sandbox/run_code/std_data/standard_data_soft_threshold.pkl']
    
    # Analyze data paths to determine test strategy
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
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args: {len(outer_args)} positional arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        try:
            agent_operator = soft_threshold(*outer_args, **outer_kwargs)
            print(f"Created operator: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Created operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Executed operator successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        try:
            result = soft_threshold(*outer_args, **outer_kwargs)
            print(f"Function executed successfully, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()