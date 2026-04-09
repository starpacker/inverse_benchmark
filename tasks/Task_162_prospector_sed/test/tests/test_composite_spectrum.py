import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_composite_spectrum import composite_spectrum
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/prospector_sed_sandbox_sandbox/run_code/std_data/standard_data_composite_spectrum.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_composite_spectrum.pkl':
            outer_path = path
    
    # Check if outer_path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_composite_spectrum.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Loaded outer data: func_name={outer_data.get('func_name')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = composite_spectrum(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute composite_spectrum: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern - Scenario B)
    if callable(result) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern (Scenario B)")
        agent_operator = result
        
        # Process inner data files
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            print(f"Loaded inner data: func_name={inner_data.get('func_name')}")
            
            # Execute the operator
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {inner_path}")
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern (Scenario A)")
        
        # Compare results directly
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

if __name__ == "__main__":
    main()