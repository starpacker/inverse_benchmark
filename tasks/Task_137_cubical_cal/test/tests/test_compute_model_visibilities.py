import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_model_visibilities import compute_model_visibilities

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/cubical_cal_sandbox_sandbox/run_code/std_data/standard_data_compute_model_visibilities.pkl'
    ]

    # Phase 0: Classify data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_model_visibilities.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_compute_model_visibilities.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data
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
    outer_output = outer_data.get('output', None)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Reconstruct operator
        try:
            agent_operator = compute_model_visibilities(*outer_args, **outer_kwargs)
            print(f"  Operator created, type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from outer data: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Execute with inner data
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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Verify
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function call")

        try:
            result = compute_model_visibilities(*outer_args, **outer_kwargs)
            print(f"  Function executed, result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Verify
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
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