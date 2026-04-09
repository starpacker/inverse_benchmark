import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_compute_rmse import compute_rmse
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data/standard_data_compute_rmse.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_rmse.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_compute_rmse.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data
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

    # Phase 2: Execute function
    try:
        agent_result = compute_rmse(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
    except Exception as e:
        print(f"FAIL: compute_rmse raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern).")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from compute_rmse, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Loaded inner data from: {inner_path}")
                print(f"  inner func_name: {inner_data.get('func_name', 'N/A')}")
                print(f"  inner args count: {len(inner_args)}, inner kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner execution successful. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    print(f"  Expected: {expected}")
                    print(f"  Got: {result}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for {os.path.basename(inner_path)}: {msg}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A (Simple function).")
        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
                sys.exit(1)
            else:
                print(f"  Verification passed: {msg}")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()