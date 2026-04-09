import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_clean_algorithm import clean_algorithm

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/data/yjh/ehtim_imaging_sandbox_sandbox/run_code/std_data/standard_data_clean_algorithm.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_clean_algorithm.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_clean_algorithm.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
        print(f"[INFO] func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        # Run outer function to get operator
        try:
            agent_operator = clean_algorithm(*outer_args, **outer_kwargs)
            print(f"[INFO] Outer function returned type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            # Execute operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner execution returned type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error executing operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Verification passed for {inner_path}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call")

        # Run the function
        try:
            result = clean_algorithm(*outer_args, **outer_kwargs)
            print(f"[INFO] Function returned type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error running clean_algorithm: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"[INFO] Verification passed")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()