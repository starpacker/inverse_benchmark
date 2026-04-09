import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_try_pymcr import try_pymcr
from verification_utils import recursive_check

def main():
    """Main test function for try_pymcr."""
    
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_try_pymcr.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_try_pymcr.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_try_pymcr.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing try_pymcr with outer args/kwargs...")
        result = try_pymcr(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing try_pymcr: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable
        if not callable(result):
            print("WARNING: Result is not callable, but inner data exists.")
            print("Falling back to direct comparison with outer output.")
            expected = outer_output
        else:
            # Load inner data and execute the operator
            try:
                inner_path = inner_paths[0]  # Use first inner path
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args/kwargs...")
                result = result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR in inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern (no inner data)")
        expected = outer_output
    
    # Phase 3: Verification
    try:
        print("Running verification...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()