import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/pyDRTtools_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    try:
        # Analyze data paths to determine test strategy
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_forward_operator.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and run the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the target function
        print("Executing forward_operator...")
        result = forward_operator(*outer_args, **outer_kwargs)
        
        # Check if this is Scenario B (factory/closure pattern)
        if inner_paths:
            # Scenario B: The result should be callable
            print("Scenario B detected: Factory/Closure pattern")
            
            if not callable(result):
                print(f"ERROR: Expected callable result but got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            # Load inner data and execute
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            print("Executing agent_operator...")
            result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple function
            print("Scenario A detected: Simple function")
        
        # Phase 2: Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, result)
        
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