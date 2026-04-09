import sys
import os
import dill
import numpy as np
import traceback

# Add the path to find the module
sys.path.insert(0, '/home/yjh/pat_sandbox/run_code')

from agent_perform_spectral_unmixing import perform_spectral_unmixing
from verification_utils import recursive_check

def main():
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_perform_spectral_unmixing.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_perform_spectral_unmixing.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_perform_spectral_unmixing.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print("Executing perform_spectral_unmixing with outer args...")
        result = perform_spectral_unmixing(*outer_args, **outer_kwargs)
        
        # Check if this is Scenario B (factory pattern) - result is callable
        if inner_paths and callable(result):
            # Scenario B: Factory/Closure Pattern
            agent_operator = result
            print(f"Function returned a callable operator. Processing {len(inner_paths)} inner data file(s)...")
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print("Executing operator with inner args...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                print("Verifying inner results...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"TEST FAILED for inner data {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            
            print("TEST PASSED")
            sys.exit(0)
        else:
            # Scenario A: Simple Function - result is the output
            print("Scenario A: Simple function execution")
            print("Verifying results...")
            
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()