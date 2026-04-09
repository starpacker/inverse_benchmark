import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/adapt_constitutive_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # -------------------------------------------------------------------------
    # Step 1: Classify data files into outer (direct) and inner (parent/closure)
    # -------------------------------------------------------------------------
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: Load outer data
    # -------------------------------------------------------------------------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # -------------------------------------------------------------------------
    # Step 3: Determine scenario
    # -------------------------------------------------------------------------
    if len(inner_paths) > 0:
        # ----- Scenario B: Factory / Closure Pattern -----
        print("[INFO] Scenario B detected: Factory/Closure pattern.")

        # Phase 1: Reconstruct the operator
        try:
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"[INFO] Operator created. Type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error calling load_and_preprocess_data with outer args: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from outer call, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data and verify
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing operator with inner args from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check for {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}.")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"[INFO] Inner test passed for: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # ----- Scenario A: Simple Function -----
        print("[INFO] Scenario A detected: Simple function call.")

        # Phase 1: Execute function with outer args
        try:
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"[INFO] Function executed. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error calling load_and_preprocess_data with outer args: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Phase 2: Verify
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
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