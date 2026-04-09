import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_adjoint_operator import adjoint_operator
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_adjoint_operator.pkl']
    
    # Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_adjoint_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_adjoint_operator.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Executing adjoint_operator with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        
        # Execute the target function
        result = adjoint_operator(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine if this is a factory pattern or simple function
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("Detected factory/closure pattern - checking if result is callable")
            
            if not callable(result):
                print("ERROR: Expected callable result for factory pattern but got non-callable")
                sys.exit(1)
            
            agent_operator = result
            
            # Load inner data and execute
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Executing agent_operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple Function
            print("Detected simple function pattern")
            expected = outer_output
        
        # Phase 3: Verification
        print("Running verification...")
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