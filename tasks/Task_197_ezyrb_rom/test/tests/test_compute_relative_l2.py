import sys
import os
import dill
import numpy as np
import traceback

from agent_compute_relative_l2 import compute_relative_l2
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_compute_relative_l2.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_relative_l2.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print("FAIL: Could not load outer data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = compute_relative_l2(*outer_args, **outer_kwargs)
        except Exception as e:
            print("FAIL: Could not create operator from outer data: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Expected callable operator from compute_relative_l2, got " + str(type(agent_operator)))
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print("FAIL: Could not load inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print("FAIL: Could not execute operator with inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print("FAIL: recursive_check raised an exception: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print("FAIL: " + str(msg))
                sys.exit(1)

    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')

        try:
            result = compute_relative_l2(*outer_args, **outer_kwargs)
        except Exception as e:
            print("FAIL: Could not execute compute_relative_l2: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print("FAIL: recursive_check raised an exception: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print("FAIL: " + str(msg))
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()