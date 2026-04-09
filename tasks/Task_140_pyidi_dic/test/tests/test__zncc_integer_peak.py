import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__zncc_integer_peak import _zncc_integer_peak
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data__zncc_integer_peak.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__zncc_integer_peak.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__zncc_integer_peak.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = _zncc_integer_peak(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _zncc_integer_peak with outer args")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) and we have inner data
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure Pattern
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()