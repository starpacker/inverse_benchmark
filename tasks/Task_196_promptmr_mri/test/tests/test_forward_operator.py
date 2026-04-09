import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    # Step 1: Classify data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (standard_data_forward_operator.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Execute the function
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator(*outer_args, **outer_kwargs)")
    except Exception as e:
        print(f"FAIL: forward_operator execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(result):
            print(f"FAIL: Expected callable from forward_operator, got {type(result)}")
            sys.exit(1)

        agent_operator = result

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Loaded inner data from: {inner_path}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
                print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator(*inner_args, **inner_kwargs)")
            except Exception as e:
                print(f"FAIL: agent_operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data verification succeeded for {inner_path}")

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: No inner data files. Comparing direct output.")
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed for outer data.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("PASS: Outer data verification succeeded.")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()