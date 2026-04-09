import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_normalize import normalize
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/abtem_sim_sandbox_sandbox/run_code/std_data/standard_data_normalize.pkl']

def main():
    try:
        # Classify data files
        outer_path = None
        inner_path = None
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_path = path
            elif basename == 'standard_data_normalize.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_normalize.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("Executing normalize function...")
        agent_result = normalize(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine scenario and verify
        if inner_path is not None:
            # Scenario B: Factory/Closure pattern
            print(f"Scenario B detected: Loading inner data from {inner_path}")
            
            if not callable(agent_result):
                print(f"ERROR: Expected callable from normalize, got {type(agent_result)}")
                sys.exit(1)
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print("Executing inner function (agent_result)...")
            result = agent_result(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple function
            print("Scenario A detected: Simple function call")
            result = agent_result
            expected = outer_output
        
        # Comparison
        print("Comparing results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()