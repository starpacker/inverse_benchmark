import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_source_time_function import source_time_function
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/mcmtpy_moment_sandbox_sandbox/run_code/std_data/standard_data_source_time_function.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data['args']
        outer_kwargs = outer_data['kwargs']
        outer_output = outer_data['output']
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        try:
            agent_operator = source_time_function(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not create operator from outer data: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Result of source_time_function is not callable but inner data exists.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data['args']
                inner_kwargs = inner_data['kwargs']
                expected = inner_data['output']
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        try:
            result = source_time_function(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not execute source_time_function: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("PASS: Outer data verified successfully.")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()