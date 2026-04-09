import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_build_greens_matrix import build_greens_matrix
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/mudpy_fault_sandbox_sandbox/run_code/std_data/standard_data_build_greens_matrix.pkl'
    ]

    # Step 1: Classify paths into outer (direct function data) and inner (parent/closure data)
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_build_greens_matrix.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_greens_matrix.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Step 3: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = build_greens_matrix(*outer_args, **outer_kwargs)
            print(f"Phase 1: build_greens_matrix returned type={type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to execute build_greens_matrix (Phase 1): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: Expected callable from build_greens_matrix, but got non-callable.")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: operator returned type={type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator (Phase 2): {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function call")

        # Execute the function
        try:
            result = build_greens_matrix(*outer_args, **outer_kwargs)
            print(f"build_greens_matrix returned type={type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute build_greens_matrix: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()