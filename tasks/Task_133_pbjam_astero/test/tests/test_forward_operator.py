import sys
import os
import dill
import numpy as np
import traceback

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pbjam_astero_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data['args']
        outer_kwargs = outer_data['kwargs']
        outer_output = outer_data['output']
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: forward_operator did not return a callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data['args']
                inner_kwargs = inner_data['kwargs']
                expected = inner_data['output']
            except Exception as e:
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR executing forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()