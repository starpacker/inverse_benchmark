import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_solve_inverse_em import solve_inverse_em

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_solve_inverse_em.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_solve_inverse_em.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find standard_data_solve_inverse_em.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = solve_inverse_em(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute solve_inverse_em with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        # The result of outer call should be callable
        if not callable(agent_result):
            print("ERROR: Expected solve_inverse_em to return a callable (Scenario B), but it did not.")
            sys.exit(1)

        agent_operator = agent_result

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
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED for inner data {inner_path}: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
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