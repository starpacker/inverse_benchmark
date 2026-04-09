import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

from agent_scan_convert import scan_convert
from verification_utils import recursive_check

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_scan_convert.pkl']
    
    # Identify files
    # In this specific case, scan_convert seems to be a standard function, not returning a closure,
    # based on the provided reference code in the prompt which returns (image_sC, znew, xnew).
    # However, the prompt mentions "Scenario B: Factory/Closure Pattern" as a priority to check.
    # We must handle both.
    
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif 'scan_convert.pkl' in p:
            outer_path = p
            
    if not outer_path:
        print("Error: standard_data_scan_convert.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from {outer_path}")
    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"Failed to load data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Execution
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print("Executing scan_convert with loaded arguments...")
        actual_result = scan_convert(*args, **kwargs)
        
        # Check if the result is a closure (Scenario B check)
        # If the function returns a callable and we have inner paths, it's a factory.
        # But looking at the reference code provided: 
        # def scan_convert(data, xb, zb): ... return image_sC, znew, xnew
        # It returns a tuple, not a callable. So this is likely Scenario A.
        # However, to be robust, we implement logic:
        
        if callable(actual_result) and inner_paths:
            print("Detected closure pattern. Executing inner functions...")
            # Scenario B: Factory/Closure
            operator = actual_result
            
            for inner_p in inner_paths:
                print(f"  Processing inner data: {inner_p}")
                inner_data = load_data(inner_p)
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                i_expected = inner_data.get('output')
                
                i_actual = operator(*i_args, **i_kwargs)
                
                passed, msg = recursive_check(i_expected, i_actual)
                if not passed:
                    print(f"FAILURE in inner function execution for {inner_p}")
                    print(msg)
                    sys.exit(1)
        else:
            # Scenario A: Standard Function
            print("Standard function execution completed. Verifying results...")
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print("FAILURE: Output mismatch.")
                print(msg)
                # Debug info
                if isinstance(actual_result, tuple) and isinstance(expected_output, tuple):
                    print(f"Result len: {len(actual_result)}, Expected len: {len(expected_output)}")
                sys.exit(1)

    except Exception as e:
        print(f"An error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()