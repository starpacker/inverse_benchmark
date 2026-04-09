import sys
import os
import dill
import numpy as np
import traceback
from agent_arange2 import arange2
from verification_utils import recursive_check

# List of data paths provided
data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_arange2.pkl']

def main():
    try:
        # 1. Identify Data Files
        outer_path = None
        inner_paths = []
        
        for p in data_paths:
            if 'standard_data_arange2.pkl' in p:
                outer_path = p
            elif 'parent_function_arange2' in p:
                inner_paths.append(p)

        if not outer_path:
            print("Error: standard_data_arange2.pkl not found in data_paths.")
            sys.exit(1)

        # 2. Load Outer Data
        print(f"Loading data from {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)

        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output')

        # 3. Execute Function
        print(f"Executing arange2 with args: {outer_args} kwargs: {outer_kwargs}")
        try:
            actual_result = arange2(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # 4. Verify Result
        # Scenario A: Simple Function (No inner paths, so just compare direct output)
        if not inner_paths:
            passed, msg = recursive_check(expected_outer_output, actual_result)
            if not passed:
                print(f"Verification FAILED: {msg}")
                sys.exit(1)
            else:
                print("Verification PASSED")
                sys.exit(0)

        # Scenario B: Factory/Closure (If inner paths existed)
        else:
            # If we are here, actual_result is likely a function (the operator)
            # We need to test it against the inner data files
            if not callable(actual_result):
                print(f"Error: Expected callable output for factory pattern, got {type(actual_result)}")
                sys.exit(1)

            print(f"Testing {len(inner_paths)} inner execution scenarios...")
            for inner_path in inner_paths:
                print(f"Loading inner data from {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')

                try:
                    inner_actual = actual_result(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"Inner execution failed for {inner_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                passed, msg = recursive_check(expected_inner_output, inner_actual)
                if not passed:
                    print(f"Inner Verification FAILED for {inner_path}: {msg}")
                    sys.exit(1)

            print("All inner verifications PASSED")
            sys.exit(0)

    except Exception as e:
        print(f"Test script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()