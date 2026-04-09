import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_add_noise import add_noise
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/empymod_sandbox_sandbox/run_code/std_data/standard_data_add_noise.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_add_noise.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_add_noise.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Fix random seed to match the data generation
        np.random.seed(42)
        
        # Execute the function
        result = add_noise(*outer_args, **outer_kwargs)
        
        # Check if this is a factory pattern (result is callable)
        if inner_paths and callable(result):
            # Scenario B: Factory/Closure pattern
            print("Detected factory/closure pattern")
            agent_operator = result
            
            # Load inner data
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            # Execute the operator
            result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple function
            print("Detected simple function pattern")
        
        # Phase 2: Verification
        print("Comparing results...")
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