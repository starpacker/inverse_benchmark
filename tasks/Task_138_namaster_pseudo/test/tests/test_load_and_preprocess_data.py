import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/namaster_pseudo_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Step 1: Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Step 3: Execute the function
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"FAIL: Error executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B: Factory/Closure pattern detected")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"FAIL: Error executing inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner call.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner call verification PASSED")

    else:
        # Scenario A: Simple function
        print("Scenario A: Simple function pattern detected")

        expected = outer_data.get('output')
        actual_result = agent_result

        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()