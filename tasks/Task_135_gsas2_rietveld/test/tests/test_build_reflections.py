import sys
import os
import dill
import traceback
import numpy as np

# Ensure the module path is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_build_reflections import build_reflections
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/gsas2_rietveld_sandbox_sandbox/run_code/std_data/standard_data_build_reflections.pkl'
    ]

    # Step 1: Classify data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (standard_data_build_reflections.pkl) found.")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Step 3: Run the function to get the operator/result
    try:
        agent_operator = build_reflections(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: Error executing build_reflections: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from build_reflections, got {type(agent_operator)}")
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
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner execution succeeded. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error executing inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function - the result IS the output
        print("Scenario A detected: Simple function call.")
        expected = outer_data.get('output')
        result = agent_operator

        try:
            passed, msg = recursive_check(expected, result)
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