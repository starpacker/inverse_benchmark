import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_euler_bernoulli_element import euler_bernoulli_element

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code/std_data/standard_data_euler_bernoulli_element.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_euler_bernoulli_element.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_euler_bernoulli_element.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = euler_bernoulli_element(*outer_args, **outer_kwargs)
        print("Phase 1: euler_bernoulli_element executed successfully.")
    except Exception as e:
        print(f"FAIL: euler_bernoulli_element raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from euler_bernoulli_element, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner execution [{idx}] completed successfully.")
            except Exception as e:
                print(f"FAIL: Inner execution [{idx}] raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception for inner [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"  Inner verification [{idx}] passed.")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: No inner data files. Comparing direct output.")

        expected = outer_data.get('output')
        actual_result = agent_result

        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()