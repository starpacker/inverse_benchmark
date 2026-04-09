import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/bxa_xray_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    # Separate outer (direct) and inner (parent_function / closure) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for forward_operator.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1: Load outer data and call forward_operator
    # ------------------------------------------------------------------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')

    try:
        agent_result = forward_operator(*outer_args, **outer_kwargs)
        print("Phase 1: forward_operator executed successfully.")
    except Exception as e:
        print(f"FAIL: forward_operator raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Determine scenario and verify
    # ------------------------------------------------------------------
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"FAIL: Expected forward_operator to return a callable (closure), "
                  f"but got {type(agent_result)}")
            sys.exit(1)

        print(f"Scenario B detected. {len(inner_paths)} inner data file(s) found.")

        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner call [{idx}] executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner call [{idx}] raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Inner call [{idx}] output mismatch.")
                print(f"  Details: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner call [{idx}] PASSED.")

    else:
        # Scenario A: Simple function call
        print("Scenario A detected. Comparing direct output.")
        expected = outer_output
        actual = agent_result

        try:
            passed, msg = recursive_check(expected, actual)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Output mismatch.")
            print(f"  Details: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()