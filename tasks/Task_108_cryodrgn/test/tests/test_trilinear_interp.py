import sys
import os
import dill
import numpy as np
import traceback

from agent_trilinear_interp import trilinear_interp
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/cryodrgn_sandbox_sandbox/run_code/std_data/standard_data_trilinear_interp.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data['args']
        outer_kwargs = outer_data['kwargs']
        expected_output = outer_data['output']
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        result = trilinear_interp(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute trilinear_interp: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check if this is Scenario B (factory pattern) or Scenario A (simple function)
    if inner_paths:
        # Scenario B: result should be callable
        if not callable(result):
            print("ERROR: Expected callable from trilinear_interp but got non-callable.")
            sys.exit(1)

        agent_operator = result
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data['args']
                inner_kwargs = inner_data['kwargs']
                expected = inner_data['output']
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: simple function, compare result directly
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()