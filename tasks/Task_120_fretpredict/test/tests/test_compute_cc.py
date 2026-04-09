import sys
import os
import dill
import numpy as np
import traceback

from agent_compute_cc import compute_cc
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_compute_cc.pkl']

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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Phase 1: Run compute_cc
    try:
        agent_result = compute_cc(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute compute_cc: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_result):
            print(f"ERROR: Expected callable from compute_cc, got {type(agent_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED for {inner_path}: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()