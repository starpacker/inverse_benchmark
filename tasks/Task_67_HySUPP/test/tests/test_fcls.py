import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_fcls import fcls
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_fcls.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_fcls.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_fcls.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Executing fcls with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        result = fcls(*outer_args, **outer_kwargs)
        
        # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print("Detected Scenario B: Factory/Closure pattern")
            
            # The result should be callable (an operator)
            if not callable(result):
                print(f"ERROR: Expected callable operator, got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            # Load inner data and execute operator
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Executing operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple function
            print("Detected Scenario A: Simple function")
            expected = outer_output
        
        # Phase 2: Verification
        print("Comparing results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()